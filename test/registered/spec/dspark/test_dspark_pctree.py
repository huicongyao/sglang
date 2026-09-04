import unittest

import torch

from sglang.srt.models.dspark import VanillaMarkov, run_markov_block
from sglang.srt.speculative.dspark_components.dspark_pctree import (
    PCTreeConfig,
    PCTreeResult,
    build_ancestor_mask,
    build_pctree,
    build_retrieve_links,
    tree_positions,
)
from sglang.srt.speculative.dspark_components.dspark_pctree_verify import (
    build_tree_mask,
    commit_accept_path_kv,
    gather_accepted_hidden,
    tree_greedy_accept,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

VOCAB = 64
RANK = 8
BLOCK = 5


def _head(seed: int = 0) -> VanillaMarkov:
    torch.manual_seed(seed)
    head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK)
    # The untrained default init gives a near-zero bias, which would make every
    # parent's ranking identical and hide parent-conditioning bugs.
    with torch.no_grad():
        head.markov_w1.weight.normal_(0.0, 1.0)
        head.markov_w2.weight.normal_(0.0, 1.0)
    return head.requires_grad_(False)


def _base_logits(bs: int = 3, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(bs, BLOCK, VOCAB)


class TestDSparkPCTree(CustomTestCase):
    def test_branching_one_reproduces_dspark_chain(self):
        """k=1 must reproduce DSpark's greedy Markov chain token-for-token.

        This is the guard on the batched parent-conditioning rewrite: any
        broadcast slip in eq.3 (wrong dim for the shared L_d, or a bias applied
        to the wrong parent) changes the argmax path and turns this red, while
        every structural test below would still pass.
        """
        head = _head()
        base_logits = _base_logits()
        anchor = torch.tensor([5, 11, 23])

        chain_tokens, _ = run_markov_block(
            head,
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=None,
            sampler=lambda logits, step: torch.argmax(logits, dim=-1),
        )

        result = build_pctree(
            base_logits=base_logits,
            anchor_tokens=anchor,
            markov_head=head,
            config=PCTreeConfig(block_size=BLOCK, branching=1, node_budget=BLOCK + 1),
        )
        self.assertTrue(torch.equal(result.tokens[:, 0], anchor))
        self.assertTrue(torch.equal(result.tokens[:, 1:], chain_tokens))
        self.assertTrue(torch.equal(result.depths[0], torch.arange(BLOCK + 1)))

    def test_batched_parents_match_per_parent_loop(self):
        """eq.3 broadcasts one shared L_d over b_d parents; assert that equals
        scoring each parent in its own call. Guards a reshape that silently
        pairs a parent with another parent's bias row."""
        head = _head(seed=2)
        base_logits = _base_logits(bs=2, seed=3)
        parents = torch.tensor([[7, 19, 31, 3], [12, 45, 2, 60]])

        batched = base_logits[:, 0, :].unsqueeze(1) + head.compute_step_bias(
            parents, None
        )
        for parent_col in range(parents.shape[1]):
            single = base_logits[:, 0, :] + head.compute_step_bias(
                parents[:, parent_col], None
            )
            torch.testing.assert_close(batched[:, parent_col, :], single)

    def test_selected_set_is_ancestor_closed(self):
        """The global top-N is taken without any closure repair, relying on
        s(c) = s(p) + log pi <= s(p). If that ordering argument is ever broken
        (e.g. scores switched to per-position instead of joint), some selected
        node loses its parent and the tree stops being verifiable."""
        head = _head(seed=4)
        result = build_pctree(
            base_logits=_base_logits(bs=4, seed=5),
            anchor_tokens=torch.tensor([1, 2, 3, 4]),
            markov_head=head,
            config=PCTreeConfig(block_size=BLOCK, branching=3, node_budget=24),
        )
        num_columns = result.tokens.shape[1]
        for row in range(result.tokens.shape[0]):
            self.assertEqual(int(result.parents[row, 0]), -1, "root must be column 0")
            for col in range(1, int(result.num_nodes[row])):
                parent = int(result.parents[row, col])
                self.assertGreaterEqual(
                    parent, 0, f"row {row} col {col} lost its parent"
                )
                self.assertLess(
                    parent, col, "parents must precede children (topological)"
                )
                self.assertEqual(
                    int(result.depths[row, col]), int(result.depths[row, parent]) + 1
                )
            self.assertLessEqual(int(result.num_nodes[row]), num_columns)

    def test_scores_are_non_increasing_along_paths(self):
        """The joint path score must be monotone non-increasing from parent to
        child -- the property the budget scheduler's greedy admission relies on."""
        result = build_pctree(
            base_logits=_base_logits(bs=2, seed=7),
            anchor_tokens=torch.tensor([9, 17]),
            markov_head=_head(seed=6),
            config=PCTreeConfig(block_size=BLOCK, branching=4, node_budget=32),
        )
        for row in range(result.tokens.shape[0]):
            for col in range(1, int(result.num_nodes[row])):
                parent = int(result.parents[row, col])
                self.assertLessEqual(
                    float(result.scores[row, col]) - 1e-5,
                    float(result.scores[row, parent]),
                )

    def test_pool_bound_and_budget_padding(self):
        """A budget larger than the reachable pool must pad rather than emit
        garbage columns; the pool bound is the paper's 1 + k + (B-1)k^2."""
        config = PCTreeConfig(block_size=3, branching=2, node_budget=64)
        self.assertEqual(config.max_pool_nodes, 1 + 2 + 2 * 4)
        result = build_pctree(
            base_logits=torch.randn(2, 3, VOCAB),
            anchor_tokens=torch.tensor([0, 1]),
            markov_head=_head(seed=8),
            config=config,
        )
        self.assertEqual(result.tokens.shape[1], 64)
        self.assertTrue(bool((result.num_nodes <= config.max_pool_nodes).all()))
        padding = result.depths < 0
        self.assertTrue(bool(padding.any()), "budget 64 > pool 11 must leave padding")
        self.assertTrue(bool(torch.isinf(result.scores[padding]).all()))

    def test_ancestor_mask_matches_parent_walk(self):
        """The mask is what the target attends through; an off-by-one in the
        parent walk would let a node see a sibling's subtree and silently
        corrupt verification instead of failing loudly."""
        result = build_pctree(
            base_logits=_base_logits(bs=2, seed=9),
            anchor_tokens=torch.tensor([4, 8]),
            markov_head=_head(seed=10),
            config=PCTreeConfig(block_size=BLOCK, branching=3, node_budget=20),
        )
        mask = build_ancestor_mask(result)
        for row in range(result.tokens.shape[0]):
            for col in range(int(result.num_nodes[row])):
                expected = {col}
                walker = int(result.parents[row, col])
                while walker >= 0:
                    expected.add(walker)
                    walker = int(result.parents[row, walker])
                got = set(mask[row, col].nonzero().flatten().tolist())
                self.assertEqual(got, expected, f"row {row} col {col}")

    def test_retrieve_links_enumerate_every_node_once(self):
        """next_token/next_sibling must cover each node exactly once; a dropped
        sibling link would make the accept walk silently skip a candidate path."""
        result = build_pctree(
            base_logits=_base_logits(bs=1, seed=11),
            anchor_tokens=torch.tensor([2]),
            markov_head=_head(seed=12),
            config=PCTreeConfig(block_size=BLOCK, branching=3, node_budget=20),
        )
        next_token, next_sibling = build_retrieve_links(result)
        visited = []
        stack = [0]
        while stack:
            node = stack.pop()
            visited.append(node)
            child = int(next_token[0, node])
            while child >= 0:
                stack.append(child)
                child = int(next_sibling[0, child])
        self.assertEqual(sorted(visited), list(range(int(result.num_nodes[0]))))

    def test_positions_share_depth_across_siblings(self):
        result = build_pctree(
            base_logits=_base_logits(bs=2, seed=13),
            anchor_tokens=torch.tensor([3, 6]),
            markov_head=_head(seed=14),
            config=PCTreeConfig(block_size=BLOCK, branching=2, node_budget=16),
        )
        positions = tree_positions(result, torch.tensor([100, 250]))
        self.assertTrue(torch.equal(positions[:, 0], torch.tensor([100, 250])))
        for row in range(2):
            for col in range(int(result.num_nodes[row])):
                self.assertEqual(
                    int(positions[row, col]),
                    (100 if row == 0 else 250) + int(result.depths[row, col]),
                )


class TestPCTreeVerify(CustomTestCase):
    """Tree verification: greedy accept, KV commit, hidden reorder."""

    def _tree(self) -> PCTreeResult:
        #        root(0)
        #        /     \
        #      a(1)    b(2)
        #      /  \      \
        #    c(3) d(4)   e(5)
        #          |
        #         f(6)
        tokens = torch.tensor([[100, 11, 12, 21, 22, 23, 31]])
        parents = torch.tensor([[-1, 0, 0, 1, 1, 2, 4]])
        depths = torch.tensor([[0, 1, 1, 2, 2, 2, 3]])
        return PCTreeResult(
            tokens=tokens,
            parents=parents,
            depths=depths,
            scores=torch.zeros(1, 7),
            num_nodes=torch.tensor([7]),
        )

    def _logits_for(self, predictions: list[int], vocab: int = 256) -> torch.Tensor:
        logits = torch.zeros(len(predictions), vocab)
        for column, token in enumerate(predictions):
            logits[column, token] = 10.0
        return logits

    def test_accept_follows_deepest_matching_path(self):
        """Greedy tree verify must follow the child whose token equals the
        parent's argmax, across siblings. A bug that only ever inspects the
        first child (i.e. treats the tree as a chain) accepts 1 here instead
        of 3, so this is the guard that the tree is really being verified."""
        tree = self._tree()
        # root -> 11 (col1), col1 -> 22 (col4), col4 -> 31 (col6), col6 -> 77 bonus
        predictions = [11, 22, 0, 0, 31, 0, 77]
        accept = tree_greedy_accept(
            tree=tree, target_logits=self._logits_for(predictions), max_depth=3
        )
        self.assertEqual(int(accept.correct_len[0]), 3)
        self.assertEqual(accept.accept_cols[0, :4].tolist(), [0, 1, 4, 6])
        self.assertEqual(int(accept.bonus[0]), 77)
        self.assertEqual(accept.out_tokens[0, :4].tolist(), [11, 22, 31, 77])

    def test_accept_takes_the_second_sibling(self):
        """The accepted child may be the *later* sibling; picking by column
        order rather than by token match would silently accept a wrong token."""
        tree = self._tree()
        # root -> 12 (col2), col2 -> 23 (col5), col5 -> 55 bonus
        predictions = [12, 0, 23, 0, 0, 55, 0]
        accept = tree_greedy_accept(
            tree=tree, target_logits=self._logits_for(predictions), max_depth=3
        )
        self.assertEqual(int(accept.correct_len[0]), 2)
        self.assertEqual(accept.accept_cols[0, :3].tolist(), [0, 2, 5])
        self.assertEqual(accept.out_tokens[0, :3].tolist(), [12, 23, 55])

    def test_accept_stops_when_no_child_matches(self):
        tree = self._tree()
        predictions = [199, 0, 0, 0, 0, 0, 0]
        accept = tree_greedy_accept(
            tree=tree, target_logits=self._logits_for(predictions), max_depth=3
        )
        self.assertEqual(int(accept.correct_len[0]), 0)
        self.assertEqual(int(accept.bonus[0]), 199)
        self.assertEqual(int(accept.out_tokens[0, 0]), 199)

    def test_commit_permutes_slots_without_duplicating(self):
        """req_to_token must end up a *permutation* of the allocated slots.

        Overwriting instead of swapping would leave the dropped column's slot
        listed twice -- once inside the committed prefix and once in the region
        the next round reallocates -- which silently corrupts a committed token.
        """
        num_columns = 7
        prefix = 5
        req_to_token = torch.zeros(1, prefix + num_columns, dtype=torch.int64)
        slots = torch.arange(1000, 1000 + prefix + num_columns, dtype=torch.int64)
        req_to_token[0] = slots
        before = req_to_token[0, prefix:].clone()

        accept_cols = torch.tensor([[0, 1, 4, 6, 0, 0, 0]])
        commit_accept_path_kv(
            req_to_token=req_to_token,
            req_pool_indices=torch.tensor([0]),
            prefix_lens=torch.tensor([prefix]),
            accept_cols=accept_cols,
            correct_len=torch.tensor([3]),
            max_depth=3,
        )
        window = req_to_token[0, prefix:]
        # Accepted depth j must address the slot the target wrote for column c_j.
        self.assertEqual(int(window[1]), int(before[1]))
        self.assertEqual(int(window[2]), int(before[4]))
        self.assertEqual(int(window[3]), int(before[6]))
        self.assertEqual(int(window[0]), int(before[0]), "root stays in place")
        self.assertEqual(
            sorted(window.tolist()), sorted(before.tolist()), "must be a permutation"
        )
        self.assertTrue(
            torch.equal(req_to_token[0, :prefix], slots[:prefix]),
            "the committed prefix must not move",
        )

    def test_hidden_reorder_matches_accept_path(self):
        hidden = torch.arange(7 * 3, dtype=torch.float32).view(1, 7, 3)
        accept_cols = torch.tensor([[0, 2, 5, 0, 0, 0, 0]])
        reordered = gather_accepted_hidden(hidden=hidden, accept_cols=accept_cols)
        self.assertTrue(torch.equal(reordered[0, 0], hidden[0, 0]))
        self.assertTrue(torch.equal(reordered[0, 1], hidden[0, 2]))
        self.assertTrue(torch.equal(reordered[0, 2], hidden[0, 5]))

    def test_mask_prefix_is_open_and_tree_block_is_ancestor_only(self):
        """The flat FULL_MASK layout triton consumes is per-request
        N x (prefix_len + N); a wrong stride here misaligns every request after
        the first, which shows up as garbage output rather than a crash."""
        tree = self._tree()
        prefix_lens = [4]
        mask = build_tree_mask(tree=tree, prefix_lens_cpu=prefix_lens)
        num_columns = tree.tokens.shape[1]
        self.assertEqual(mask.numel(), num_columns * (prefix_lens[0] + num_columns))
        rows = mask.view(num_columns, prefix_lens[0] + num_columns)
        self.assertTrue(bool(rows[:, : prefix_lens[0]].all()), "prefix must be visible")
        tree_block = rows[:, prefix_lens[0] :]
        # col 6's ancestors are 4, 1, 0.
        self.assertEqual(
            sorted(tree_block[6].nonzero().flatten().tolist()), [0, 1, 4, 6]
        )
        self.assertEqual(tree_block[2].nonzero().flatten().tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()

