"""Offline PCTree-vs-DSpark acceptance evaluation.

Records, for each draft round, the DSpark chain and the PCTree tree built from
the *same* base logits, so the two can be scored against the target's own greedy
continuation afterwards.

Why this measures the paper's tau without any tree attention: under greedy target
verification a draft token is accepted iff it equals the target's argmax given
the accepted prefix. So the accepted length of a candidate set is the length of
the longest prefix of the target's greedy continuation that appears as a
root-to-node path. Running the (lossless) DSpark server greedily already emits
exactly that continuation, so the committed token stream is the ground truth.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_pctree import (
    PCTreeConfig,
    build_pctree,
)

logger = logging.getLogger(__name__)


class PCTreeEvalRecorder:
    """Appends one jsonl record per (round, request). Diagnostic-only path.

    Records one tree per configured branching factor so a single server run
    yields the whole k-sweep; k=1 doubles as the correctness gate (its tree must
    score identically to the chain).
    """

    def __init__(self, *, path: str, configs: list[PCTreeConfig]) -> None:
        self._path = path
        self._configs = configs
        self._lock = threading.Lock()
        self._round = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "header",
                        "block_size": configs[0].block_size,
                        "branchings": [c.branching for c in configs],
                        "node_budget": configs[0].node_budget,
                    }
                )
                + "\n"
            )
        logger.info(
            "DSpark PCTree eval recording to %s (k=%s, N=%d).",
            path,
            [c.branching for c in configs],
            configs[0].node_budget,
        )

    def record(
        self,
        *,
        base_logits: torch.Tensor,
        anchor_tokens: torch.Tensor,
        chain_tokens: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_ids: list[str],
        markov_head,
        hidden_states: Optional[torch.Tensor],
    ) -> None:
        trees = {}
        for config in self._configs:
            tree = build_pctree(
                base_logits=base_logits,
                anchor_tokens=anchor_tokens,
                markov_head=markov_head,
                config=config,
                hidden_states=hidden_states,
            )
            trees[config.branching] = (
                tree.tokens.tolist(),
                tree.parents.tolist(),
                tree.num_nodes.tolist(),
                tree.scores.tolist(),
            )
        anchors = anchor_tokens.tolist()
        prefixes = prefix_lens.tolist()
        chains = chain_tokens.tolist()
        with self._lock:
            round_index = self._round
            self._round += 1
            with open(self._path, "a") as handle:
                for row, req_id in enumerate(req_ids):
                    handle.write(
                        json.dumps(
                            {
                                "kind": "round",
                                "round": round_index,
                                "rid": req_id,
                                "prefix_len": int(prefixes[row]),
                                "anchor": int(anchors[row]),
                                "chain": chains[row],
                                "trees": {
                                    str(branching): {
                                        "tokens": tokens[row],
                                        "parents": parents[row],
                                        "num_nodes": int(counts[row]),
                                        "scores": scores[row],
                                    }
                                    for branching, (
                                        tokens,
                                        parents,
                                        counts,
                                        scores,
                                    ) in trees.items()
                                },
                            }
                        )
                        + "\n"
                    )


_RECORDER: Optional[PCTreeEvalRecorder] = None
_RESOLVED = False


def maybe_get_recorder(*, block_size: int) -> Optional[PCTreeEvalRecorder]:
    """Build the recorder on first use; returns None when the env var is unset."""
    global _RECORDER, _RESOLVED
    if _RESOLVED:
        return _RECORDER
    _RESOLVED = True
    path = envs.SGLANG_DSPARK_PCTREE_EVAL_PATH.get()
    if not path:
        return None
    node_budget = envs.SGLANG_DSPARK_PCTREE_NODE_BUDGET.get()
    branchings = [
        int(token)
        for token in envs.SGLANG_DSPARK_PCTREE_BRANCHING.get().split(",")
        if token.strip()
    ]
    _RECORDER = PCTreeEvalRecorder(
        path=path,
        configs=[
            PCTreeConfig(
                block_size=block_size, branching=branching, node_budget=node_budget
            )
            for branching in branchings
        ],
    )
    return _RECORDER

