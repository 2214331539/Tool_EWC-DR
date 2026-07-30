from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from toolhcl_ewcdr_pure.data import parse_split


def _row(query: str, source_id: str, tool_id: int = 0) -> dict:
    return {
        "source_id": source_id,
        "target_tool_id": tool_id,
        "conversations": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "<<tool&&api>>"},
        ],
    }


class TransactionDecontaminationTest(unittest.TestCase):
    def test_source_or_normalized_query_overlap_is_removed(self):
        rows = [
            _row("Keep this query", "train-1"),
            _row("same   QUERY", "train-2"),
            _row("Different text", "eval-source"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            tools = {0: type("Tool", (), {"tool_name": "tool", "api_name": "api"})()}
            samples, stats = parse_split(
                path,
                mapping={"apifortool": 0},
                tools=tools,
                source_stage="base",
                visible_count=1,
                target_id_field="target_tool_id",
                excluded_source_ids={"eval-source"},
                excluded_normalized_queries={"same query"},
            )

        self.assertEqual([sample.query_text for sample in samples], ["Keep this query"])
        self.assertEqual(stats["excluded_eval_query_overlap"], 1)
        self.assertEqual(stats["excluded_eval_source_overlap"], 1)
        self.assertEqual(stats["excluded_eval_source_or_query_overlap"], 2)


if __name__ == "__main__":
    unittest.main()
