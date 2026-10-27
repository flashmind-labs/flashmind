import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from harbor.models.agent.context import AgentContext
from harbor_agent import FlashmindAgent, model_parts


class HarborAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        binary = self.root / "flsh"
        binary.write_text("test binary")
        self.agent = FlashmindAgent(
            logs_dir=self.root,
            binary_path=str(binary),
            model_name="openrouter/example/model",
            extra_env={"OPENROUTER_API_KEY": "test-key"},
        )

    async def test_prompt_is_one_shell_argument_and_only_selected_credentials_are_sent(
        self,
    ):
        prompt = "Fix 'quotes', $(false), `false`, and\nnewlines."
        self.agent.exec_as_agent = AsyncMock()
        self.agent._upload_config_text = AsyncMock()
        context = AgentContext()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "unrelated-key"}):
            await self.agent.run(prompt, AsyncMock(), context)
        self.assertTrue(context.is_empty(), "Harbor must call the post-run usage hook")
        call = self.agent.exec_as_agent.call_args.kwargs
        arguments = shlex.split(call["command"])
        self.assertNotIn("--tools", arguments)
        self.assertEqual(arguments[arguments.index("--prompt") + 1], prompt)
        self.assertEqual(
            arguments[arguments.index("--model") + 1], "openrouter:example/model"
        )
        self.assertEqual(call["env"]["OPENROUTER_API_KEY"], "test-key")
        self.assertNotIn("ANTHROPIC_API_KEY", call["env"])
        config = self.agent._upload_config_text.call_args.kwargs["content"]
        self.assertNotIn("test-key", config)

    async def test_selected_tools_reach_the_cli_and_result_metadata(self):
        agent = FlashmindAgent(
            logs_dir=self.root,
            binary_path=str(self.root / "flsh"),
            model_name="openrouter/example/model",
            extra_env={"OPENROUTER_API_KEY": "test-key"},
            tools="read, write,edit,find,exec,process,read",
        )
        agent.exec_as_agent = AsyncMock()
        agent._upload_config_text = AsyncMock()
        context = AgentContext()
        await agent.run("task", AsyncMock(), context)
        arguments = shlex.split(agent.exec_as_agent.call_args.kwargs["command"])
        self.assertEqual(
            arguments[arguments.index("--tools") + 1],
            "read,write,edit,find,exec,process",
        )
        self.assertTrue(context.is_empty())
        agent.populate_context_post_run(context)
        self.assertEqual(
            context.metadata["tools"],
            ["read", "write", "edit", "find", "exec", "process"],
        )

    def test_invalid_tool_lists_are_rejected_before_running(self):
        for tools in ["", "read,", "read,,exec", "read;exec"]:
            with self.subTest(tools=tools), self.assertRaises(ValueError):
                FlashmindAgent(
                    logs_dir=self.root,
                    binary_path=str(self.root / "flsh"),
                    model_name="openrouter/example/model",
                    tools=tools,
                )

    async def test_failed_agent_execution_propagates(self):
        self.agent.exec_as_agent = AsyncMock(side_effect=[None, RuntimeError("failed")])
        self.agent._upload_config_text = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            await self.agent.run("task", AsyncMock(), AgentContext())

    def test_usage_counts_parent_and_nested_subagents_once(self):
        parent = {
            "type": "usage",
            "data": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "cache_read_tokens": 40,
            },
        }
        child = {
            "type": "spawned_event",
            "data": {
                "event": {
                    "type": "usage",
                    "data": {
                        "prompt_tokens": 50,
                        "completion_tokens": 5,
                        "cache_read_tokens": 20,
                    },
                },
            },
        }
        grandchild = {"type": "spawned_event", "data": {"event": child}}
        with (self.root / "events.jsonl").open("w") as stream:
            for event in [parent, child, grandchild, {"type": "done", "data": "ok"}]:
                stream.write(json.dumps(event) + "\n")
            stream.write('{"type":')
        (self.root / "stderr.txt").write_text("Cost: $0.0123\nResume: example\n")
        context = AgentContext()
        self.agent.populate_context_post_run(context)
        self.assertEqual(context.n_input_tokens, 200)
        self.assertEqual(context.n_output_tokens, 20)
        self.assertEqual(context.n_cache_tokens, 80)
        self.assertEqual(context.cost_usd, 0.0123)

    def test_missing_usage_is_unknown(self):
        (self.root / "events.jsonl").write_text('{"type":"error","data":"failed"}\n')
        context = AgentContext()
        self.agent.populate_context_post_run(context)
        self.assertIsNone(context.n_input_tokens)
        self.assertIsNone(context.cost_usd)

    def test_model_metadata_preserves_provider_and_full_model_id(self):
        info = self.agent.to_agent_info()
        self.assertEqual(info.model_info.provider, "openrouter")
        self.assertEqual(info.model_info.name, "example/model")

    def test_model_selectors_preserve_tags_and_vendor_prefixes(self):
        for selector, expected in [
            ("ollama/model:tag", ("ollama", "model:tag")),
            ("ollama:model:tag", ("ollama", "model:tag")),
            ("openrouter:vendor/model", ("openrouter", "vendor/model")),
            ("openrouter/vendor/model", ("openrouter", "vendor/model")),
        ]:
            with self.subTest(selector=selector):
                self.assertEqual(model_parts(selector), expected)
        for selector in ["model", "openrouter/", "unsupported/model"]:
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                model_parts(selector)


if __name__ == "__main__":
    unittest.main()
