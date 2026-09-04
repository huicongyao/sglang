"""PCTree: parent-conditioned draft-tree construction for DSpark.

Turns DSpark's single top-1 Markov chain into a fixed-budget draft tree by
re-scoring the shared block logits under every concrete parent, per
arXiv:2608.02123. Training-free: it reuses the pretrained Markov head and the
one parallel backbone forward that produced ``base_logits``.

Sequential stage count is unchanged (``block_size`` Markov calls); only the
Markov batch widens, from 1 to at most ``branching`` parents.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch


class PCTreeConfig(msgspec.Struct, frozen=True):
    """block_size = B (Markov stages), branching = k, node_budget = N incl. root."""

    block_size: int
    branching: int
    node_budget: int

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"PCTree block_size must be > 0, got {self.block_size}.")
        if self.branching <= 0:
            raise ValueError(f"PCTree branching must be > 0, got {self.branching}.")
        if self.node_budget < 1:
            raise ValueError(
                f"PCTree node_budget must be >= 1 (root counts), got {self.node_budget}."
            )

    @property
    def max_pool_nodes(self) -> int:
        """Paper's bound: root + k first-level + (B-1)k^2 from the pruned frontier."""
        return 1 + self.branching + (self.block_size - 1) * self.branching**2


class PCTreeResult(msgspec.Struct, frozen=True):
    """Column 0 of every row is the root (the anchor token), score 0, depth 0.

    ``parents`` indexes columns of this same result (-1 for the root); padding
    columns carry token 0, depth -1, score -inf and parent -1.
    """

    tokens: torch.Tensor  # [bs, N] int64
    parents: torch.Tensor  # [bs, N] int64, column index within the row
    depths: torch.Tensor  # [bs, N] int64, root = 0, padding = -1
    scores: torch.Tensor  # [bs, N] float32 joint path log-prob
    num_nodes: torch.Tensor  # [bs] int64, number of real (non-padding) columns


def _step_bias(
    markov_head,
    *,
    parent_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
) -> torch.Tensor:
    """Markov(p) for every parent in the frontier: [bs, b_d] -> [bs, b_d, V].

    ``hidden_states`` is the depth-d backbone hidden [bs, H]; it is broadcast
    across parents because all parents at a depth share the same block position.
    """
    if markov_head.markov_head_type == "rnn":
        raise NotImplementedError(
            "PCTree needs first-order parent conditioning; the RNN head carries a "
            "per-path prefix state that would have to be expanded per frontier "
            "parent. The paper follows DSpark's Markov default for this reason."
        )
    b_d = parent_tokens.shape[1]
    expanded_hidden = None
    if hidden_states is not None:
        expanded_hidden = hidden_states.unsqueeze(1).expand(-1, b_d, -1)
    return markov_head.compute_step_bias(parent_tokens, expanded_hidden)


def _expand_frontier(
    *,
    base_logits_d: torch.Tensor,
    markov_head,
    parent_tokens: torch.Tensor,
    parent_scores: torch.Tensor,
    hidden_states_d: Optional[torch.Tensor],
    branching: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Markov stage (eq.2/eq.3) plus local top-k, scored by eq.4.

    Returns (child_tokens, child_scores), both [bs, b_d, k].
    """
    bias = _step_bias(
        markov_head, parent_tokens=parent_tokens, hidden_states=hidden_states_d
    )
    # eq.3: the shared L_d is broadcast across the b_d frontier parents.
    logits = base_logits_d.unsqueeze(1) + bias
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    top_log_probs, top_tokens = torch.topk(log_probs, branching, dim=-1)
    # eq.4: s(child) = s(parent) + log pi_d(child | parent)
    return top_tokens, parent_scores.unsqueeze(-1) + top_log_probs


class _Pool:
    """Candidate accumulator. Node id == insertion index, so a child's id always
    exceeds its parent's -- that is what makes an id-ascending sort topological."""

    def __init__(self, *, anchor_tokens: torch.Tensor) -> None:
        bs = anchor_tokens.shape[0]
        device = anchor_tokens.device
        zeros = torch.zeros(bs, 1, dtype=torch.int64, device=device)
        self._tokens = [anchor_tokens.view(bs, 1).to(torch.int64)]
        self._scores = [torch.zeros(bs, 1, dtype=torch.float32, device=device)]
        self._parents = [torch.full((bs, 1), -1, dtype=torch.int64, device=device)]
        self._depths = [zeros]
        self.size = 1

    def add_layer(
        self,
        *,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        parent_ids: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        """Append a flattened [bs, m] layer; returns its node ids [bs, m]."""
        bs, m = tokens.shape
        ids = torch.arange(
            self.size, self.size + m, dtype=torch.int64, device=tokens.device
        )
        self._tokens.append(tokens.to(torch.int64))
        self._scores.append(scores.to(torch.float32))
        self._parents.append(parent_ids.to(torch.int64))
        self._depths.append(
            torch.full((bs, m), depth, dtype=torch.int64, device=tokens.device)
        )
        self.size += m
        return ids.unsqueeze(0).expand(bs, -1)

    def stack(self) -> tuple[torch.Tensor, ...]:
        return (
            torch.cat(self._tokens, dim=1),
            torch.cat(self._scores, dim=1),
            torch.cat(self._parents, dim=1),
            torch.cat(self._depths, dim=1),
        )


def _select_budget(
    *,
    pool_scores: torch.Tensor,
    node_budget: int,
) -> torch.Tensor:
    """Global top-N by (score desc, depth asc, stable id), returned id-ascending.

    The pool is laid out depth-ascending and id-ascending within a depth, so a
    *stable* descending sort on the score alone reproduces the paper's full
    tie-break. Ancestor-closure needs no repair: s(c) = s(p) + log pi <= s(p),
    so an ancestor never sorts below its descendant, and on an exact tie
    (log pi == 0) stability keeps the shallower node first.
    """
    num_selected = min(node_budget, pool_scores.shape[1])
    order = torch.argsort(pool_scores, dim=-1, descending=True, stable=True)
    # Ascending node id is a topological order (child id > parent id).
    return torch.sort(order[:, :num_selected], dim=-1).values


def _remap_to_columns(
    *,
    selected_ids: torch.Tensor,
    pool: tuple[torch.Tensor, ...],
    node_budget: int,
) -> PCTreeResult:
    """Gather the selected nodes into [bs, node_budget] columns, re-basing parent
    pointers from pool ids to column indices and padding any unused tail."""
    pool_tokens, pool_scores, pool_parents, pool_depths = pool
    bs, pool_size = pool_tokens.shape
    device = pool_tokens.device
    num_selected = selected_ids.shape[1]

    column_of_id = torch.full((bs, pool_size), -1, dtype=torch.int64, device=device)
    column_of_id.scatter_(
        1,
        selected_ids,
        torch.arange(num_selected, dtype=torch.int64, device=device).expand(bs, -1),
    )

    tokens = torch.gather(pool_tokens, 1, selected_ids)
    scores = torch.gather(pool_scores, 1, selected_ids)
    depths = torch.gather(pool_depths, 1, selected_ids)
    parent_ids = torch.gather(pool_parents, 1, selected_ids)
    # The root's -1 would index the last column; clamp then restore it.
    parents = torch.where(
        parent_ids >= 0,
        torch.gather(column_of_id, 1, parent_ids.clamp_min(0)),
        torch.full_like(parent_ids, -1),
    )

    pad = node_budget - num_selected
    if pad > 0:
        pad_long = torch.zeros(bs, pad, dtype=torch.int64, device=device)
        pad_neg = torch.full((bs, pad), -1, dtype=torch.int64, device=device)
        tokens = torch.cat([tokens, pad_long], dim=1)
        scores = torch.cat(
            [scores, torch.full((bs, pad), float("-inf"), device=device)], dim=1
        )
        depths = torch.cat([depths, pad_neg], dim=1)
        parents = torch.cat([parents, pad_neg.clone()], dim=1)

    return PCTreeResult(
        tokens=tokens,
        parents=parents,
        depths=depths,
        scores=scores,
        num_nodes=torch.full((bs,), num_selected, dtype=torch.int64, device=device),
    )


def build_pctree(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    markov_head,
    config: PCTreeConfig,
    hidden_states: Optional[torch.Tensor] = None,
) -> PCTreeResult:
    """Algorithm 1: build the budgeted parent-conditioned draft tree.

    ``base_logits`` [bs, B, V] are the shared block logits from the single
    parallel backbone forward; ``anchor_tokens`` [bs] is the last verified token
    (the tree root). ``hidden_states`` [bs, B, H] is only read by gated heads.
    """
    config.validate()
    if base_logits.dim() != 3:
        raise ValueError(
            f"base_logits must be [bs, B, V], got {tuple(base_logits.shape)}."
        )
    if base_logits.shape[1] < config.block_size:
        raise ValueError(
            f"base_logits has {base_logits.shape[1]} block positions but "
            f"block_size={config.block_size}."
        )

    pool = _Pool(anchor_tokens=anchor_tokens)
    frontier_tokens = anchor_tokens.view(-1, 1).to(torch.int64)
    frontier_scores = torch.zeros_like(frontier_tokens, dtype=torch.float32)
    frontier_ids = torch.zeros_like(frontier_tokens)

    for depth in range(config.block_size):
        child_tokens, child_scores = _expand_frontier(
            base_logits_d=base_logits[:, depth, :],
            markov_head=markov_head,
            parent_tokens=frontier_tokens,
            parent_scores=frontier_scores,
            hidden_states_d=(
                None if hidden_states is None else hidden_states[:, depth, :]
            ),
            branching=config.branching,
        )
        bs, num_parents, branching = child_tokens.shape
        flat_tokens = child_tokens.reshape(bs, num_parents * branching)
        flat_scores = child_scores.reshape(bs, num_parents * branching)
        flat_parents = (
            frontier_ids.unsqueeze(-1)
            .expand(bs, num_parents, branching)
            .reshape(bs, -1)
        )
        layer_ids = pool.add_layer(
            tokens=flat_tokens,
            scores=flat_scores,
            parent_ids=flat_parents,
            depth=depth + 1,
        )
        if depth + 1 == config.block_size:
            break
        # Frontier pruning: only the k highest-scoring nodes of this layer expand.
        keep = min(config.branching, flat_scores.shape[1])
        frontier_scores, keep_index = torch.topk(flat_scores, keep, dim=-1)
        frontier_tokens = torch.gather(flat_tokens, 1, keep_index)
        frontier_ids = torch.gather(layer_ids, 1, keep_index)

    stacked = pool.stack()
    selected_ids = _select_budget(
        pool_scores=stacked[1], node_budget=config.node_budget
    )
    return _remap_to_columns(
        selected_ids=selected_ids, pool=stacked, node_budget=config.node_budget
    )


def build_ancestor_mask(result: PCTreeResult) -> torch.Tensor:
    """Ancestor-only attention mask [bs, N, N]: mask[i, j] iff j is i or an ancestor.

    Callers prepend the verified-prefix columns (all True) themselves; this is
    only the tree block of the verify mask.
    """
    bs, num_columns = result.tokens.shape
    device = result.tokens.device
    mask = (
        torch.eye(num_columns, dtype=torch.bool, device=device)
        .expand(bs, -1, -1)
        .clone()
    )
    mask[result.depths < 0] = False
    current = result.parents.clone()
    rows = torch.arange(num_columns, device=device).view(1, -1).expand(bs, -1)
    # Depth is bounded by the number of columns, so this terminates.
    for _ in range(num_columns):
        valid = current >= 0
        if not bool(valid.any()):
            break
        batch_idx, col_idx = valid.nonzero(as_tuple=True)
        mask[batch_idx, rows[batch_idx, col_idx], current[batch_idx, col_idx]] = True
        current = torch.where(
            valid,
            torch.gather(result.parents, 1, current.clamp_min(0)),
            torch.full_like(current, -1),
        )
    return mask


def build_retrieve_links(result: PCTreeResult) -> tuple[torch.Tensor, torch.Tensor]:
    """(next_token, next_sibling) in SGLang's tree-accept convention, -1 for none.

    ``next_token[i]`` is i's first child column, ``next_sibling[i]`` the next
    column sharing i's parent. Columns are topological, so both look forward.
    """
    bs, num_columns = result.parents.shape
    device = result.parents.device
    next_token = torch.full((bs, num_columns), -1, dtype=torch.int64, device=device)
    next_sibling = torch.full((bs, num_columns), -1, dtype=torch.int64, device=device)
    parents = result.parents
    real = result.depths >= 0
    # N is the verification budget (32 in the paper); a column loop is cheap and
    # keeps the sibling ordering exactly the column ordering.
    for col in range(num_columns - 1, -1, -1):
        parent = parents[:, col]
        has_parent = (parent >= 0) & real[:, col]
        rows = has_parent.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        parent_rows = parent[rows]
        next_sibling[rows, col] = next_token[rows, parent_rows]
        next_token[rows, parent_rows] = col
    return next_token, next_sibling


def tree_positions(result: PCTreeResult, prefix_lens: torch.Tensor) -> torch.Tensor:
    """position id = tree depth + verified prefix length (paper appendix A).

    Siblings share a position while keeping distinct ancestry; padding columns
    reuse the prefix position and are masked out downstream.
    """
    depths = result.depths.clamp_min(0)
    return prefix_lens.view(-1, 1).to(depths.dtype) + depths


def chain_result_from_tokens(
    *,
    anchor_tokens: torch.Tensor,
    chain_tokens: torch.Tensor,
) -> PCTreeResult:
    """Wrap a DSpark chain as a degenerate PCTree result (for k=1 comparisons)."""
    bs, gamma = chain_tokens.shape
    device = chain_tokens.device
    tokens = torch.cat(
        [anchor_tokens.view(bs, 1).to(torch.int64), chain_tokens.to(torch.int64)], dim=1
    )
    columns = gamma + 1
    parents = (
        torch.arange(-1, gamma, dtype=torch.int64, device=device)
        .expand(bs, -1)
        .contiguous()
    )
    depths = (
        torch.arange(columns, dtype=torch.int64, device=device)
        .expand(bs, -1)
        .contiguous()
    )
    return PCTreeResult(
        tokens=tokens,
        parents=parents,
        depths=depths,
        scores=torch.zeros(bs, columns, dtype=torch.float32, device=device),
        num_nodes=torch.full((bs,), columns, dtype=torch.int64, device=device),
    )


