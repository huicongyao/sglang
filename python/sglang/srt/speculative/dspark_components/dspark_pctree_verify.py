"""Real tree verification for PCTree drafts on the DSpark path.

Implements the target-side half of arXiv:2608.02123: pack the budgeted tree into
one target forward with an ancestor-only attention mask, verify it with the
standard greedy tree rule, and commit the accepted root-to-node path.

Layout notes
------------
* Column 0 of the tree is the root (the previous round's bonus token), so the
  verify window is [root, N-1 candidate nodes] -- the same shape contract as the
  chain path's [anchor, gamma drafts].
* The mask follows SGLang's ``TreeMaskMode.FULL_MASK`` convention that
  ``triton_backend`` consumes at TARGET_VERIFY: per request, N query rows of
  (prefix_len + N) bools, prefix columns all True, tree columns ancestor-or-self.
* Accepted KV is committed by *permuting* ``req_to_token`` rather than moving KV
  bytes: entry (prefix + j) is swapped with the accepted node's column entry, so
  the row keeps exactly its allocated slot set (no duplication, no leak).
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.speculative.dspark_components.dspark_pctree import (
    PCTreeConfig,
    PCTreeResult,
    build_ancestor_mask,
    build_pctree,
    tree_positions,
)


class TreeVerifyBundle(msgspec.Struct):
    """Everything the target verify forward and the accept step need."""

    tree: PCTreeResult
    verify_ids: torch.Tensor  # [bs * N] flat draft tokens (col 0 = root)
    positions: torch.Tensor  # [bs * N] flat, depth + prefix_len
    custom_mask: torch.Tensor  # flat FULL_MASK over requests


def build_tree_mask(
    *,
    tree: PCTreeResult,
    prefix_lens_cpu: list[int],
) -> torch.Tensor:
    """Flat FULL_MASK: per request N rows of (prefix_len + N) allow-bits.

    Padding columns are masked out as keys but still emit a query row; their
    logits are never read because the accept walk only follows real nodes.
    """
    ancestor = build_ancestor_mask(tree)
    bs, num_columns, _ = ancestor.shape
    device = ancestor.device
    blocks = []
    for row in range(bs):
        prefix_len = int(prefix_lens_cpu[row])
        prefix_block = torch.ones(
            num_columns, prefix_len, dtype=torch.bool, device=device
        )
        blocks.append(torch.cat([prefix_block, ancestor[row]], dim=1).flatten())
    return torch.cat(blocks)


def build_tree_verify_bundle(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    markov_head,
    hidden_states: Optional[torch.Tensor],
    config: PCTreeConfig,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: list[int],
) -> TreeVerifyBundle:
    tree = build_pctree(
        base_logits=base_logits,
        anchor_tokens=anchor_tokens,
        markov_head=markov_head,
        config=config,
        hidden_states=hidden_states,
    )
    positions = tree_positions(tree, prefix_lens)
    return TreeVerifyBundle(
        tree=tree,
        verify_ids=tree.tokens.reshape(-1).contiguous(),
        positions=positions.reshape(-1).contiguous(),
        custom_mask=build_tree_mask(tree=tree, prefix_lens_cpu=prefix_lens_cpu),
    )


class TreeAcceptOuts(msgspec.Struct):
    accept_cols: torch.Tensor  # [bs, N] columns of the accepted path, col 0 = root
    correct_len: torch.Tensor  # [bs] accepted *draft* tokens (excludes root/bonus)
    bonus: torch.Tensor  # [bs] target token after the last accepted node
    out_tokens: torch.Tensor  # [bs, N] accepted tokens then bonus, then padding


def tree_greedy_accept(
    *,
    tree: PCTreeResult,
    target_logits: torch.Tensor,
    max_depth: int,
) -> TreeAcceptOuts:
    """Standard greedy tree verification.

    ``target_logits`` is [bs * N, V] in tree-column order; row c predicts the
    token that should follow node c. A child is accepted iff its token equals
    its parent's argmax, which makes the accepted set a single root-to-node path.
    """
    bs, num_columns = tree.tokens.shape
    device = tree.tokens.device
    logits = target_logits.view(bs, num_columns, -1)
    predicted = torch.argmax(logits, dim=-1)  # [bs, N]

    rows = torch.arange(bs, device=device)
    node = torch.zeros(bs, dtype=torch.int64, device=device)
    accept_cols = torch.zeros(bs, num_columns, dtype=torch.int64, device=device)
    correct_len = torch.zeros(bs, dtype=torch.int64, device=device)
    alive = torch.ones(bs, dtype=torch.bool, device=device)
    real = tree.depths >= 0

    for step in range(max_depth):
        target_token = predicted[rows, node]  # [bs]
        candidate = (
            (tree.parents == node.unsqueeze(1))
            & (tree.tokens == target_token.unsqueeze(1))
            & real
        )
        found = candidate.any(dim=1) & alive
        # First matching column; ties cannot happen for a single parent because
        # local top-k returns distinct tokens.
        first = torch.argmax(candidate.to(torch.int8), dim=1)
        node = torch.where(found, first, node)
        accept_cols[:, step + 1] = node
        correct_len = correct_len + found.to(correct_len.dtype)
        alive = found
        if not bool(alive.any()):
            break

    bonus = predicted[rows, node]
    out_tokens = torch.zeros(bs, num_columns, dtype=torch.int64, device=device)
    path_tokens = torch.gather(tree.tokens, 1, accept_cols)
    columns = torch.arange(num_columns, device=device).unsqueeze(0)
    # Accepted draft tokens occupy [0, correct_len); the bonus follows them.
    keep = columns < correct_len.unsqueeze(1)
    out_tokens = torch.where(keep, path_tokens.roll(-1, dims=1), out_tokens)
    out_tokens.scatter_(1, correct_len.unsqueeze(1), bonus.unsqueeze(1))
    return TreeAcceptOuts(
        accept_cols=accept_cols,
        correct_len=correct_len,
        bonus=bonus,
        out_tokens=out_tokens,
    )


def commit_accept_path_kv(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    accept_cols: torch.Tensor,
    correct_len: torch.Tensor,
    max_depth: int,
) -> None:
    """Permute req_to_token so the accepted path occupies contiguous positions.

    Entry (prefix + j) must address the KV slot the target wrote for the accepted
    node at depth j. Swapping (rather than overwriting) keeps the row's slot set a
    permutation of what the allocator handed out, so the columns we drop stay
    owned by this row and get reused by the next round instead of aliasing a
    committed token. Column indices along a path are strictly increasing, so
    ascending j never disturbs an already-fixed entry.
    """
    rows = torch.arange(req_pool_indices.shape[0], device=req_to_token.device)
    pool_rows = req_pool_indices.to(torch.int64)
    for depth in range(1, max_depth + 1):
        active = correct_len >= depth
        if not bool(active.any()):
            break
        active_rows = rows[active]
        pool = pool_rows[active_rows]
        dest = prefix_lens[active_rows].to(torch.int64) + depth
        source = prefix_lens[active_rows].to(torch.int64) + accept_cols[
            active_rows, depth
        ]
        dest_slot = req_to_token[pool, dest].clone()
        source_slot = req_to_token[pool, source].clone()
        req_to_token[pool, dest] = source_slot
        req_to_token[pool, source] = dest_slot


def gather_accepted_hidden(
    *,
    hidden: torch.Tensor,
    accept_cols: torch.Tensor,
) -> torch.Tensor:
    """Reorder [bs, N, H] verify hidden states into accepted-path order.

    The chain commit path writes hidden state j to position prefix + j gated by
    commit_lens, so the accepted path's states must land at the same strides.
    """
    bs, num_columns, hidden_size = hidden.shape
    index = accept_cols.unsqueeze(-1).expand(bs, num_columns, hidden_size)
    return torch.gather(hidden, 1, index)
