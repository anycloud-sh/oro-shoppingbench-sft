import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import ValidationError

from infer import parse_tool_calls, read_inputs

ROOT = Path(__file__).resolve().parents[1]


class InputAndOutputTests(unittest.TestCase):
    def setUp(self):
        self.rows, _ = read_inputs(ROOT / "examples/shopping.jsonl")

    def test_real_reference_outputs_satisfy_source_tool_schemas(self):
        references = [
            json.loads(line)
            for line in (ROOT / "validation/reference.jsonl").read_text().splitlines()
        ]
        self.assertEqual([r["id"] for r in references], [r["id"] for r in self.rows])
        for reference, row in zip(references, self.rows):
            calls = parse_tool_calls(reference["raw_text"], row["tools"])
            self.assertEqual(calls[0]["name"], "find_product")
            self.assertIsInstance(calls[0]["arguments"]["q"], str)

    def test_reasoning_examples_are_not_executed_as_tool_calls(self):
        text = '<think>Example: <tool_call>not json</tool_call></think><tool_call>{"name":"terminate","arguments":{}}</tool_call>'
        self.assertEqual(
            parse_tool_calls(text, self.rows[0]["tools"]),
            [{"name": "terminate", "arguments": {}}],
        )

    def test_rejects_unknown_tools_malformed_json_and_truncation(self):
        for text in [
            '<tool_call>{"name":"buy_now","arguments":{}}</tool_call>',
            "<tool_call>{broken}</tool_call>",
            '<tool_call>{"name":"terminate","arguments":{}}',
            "<think>unfinished reasoning",
            "I found a product.",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_tool_calls(text, self.rows[0]["tools"])

    def test_checks_argument_types(self):
        with self.assertRaises(ValidationError):
            parse_tool_calls(
                '<tool_call>{"name":"find_product","arguments":{"page":"one"}}</tool_call>',
                self.rows[0]["tools"],
            )

    def test_rejects_duplicate_ids_and_target_assistant_leakage(self):
        for rows in [
            [self.rows[0], self.rows[0]],
            [
                {
                    **self.rows[0],
                    "messages": self.rows[0]["messages"]
                    + [{"role": "assistant", "content": "target"}],
                }
            ],
        ]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaises(ValueError):
                    read_inputs(path)


if __name__ == "__main__":
    unittest.main()
