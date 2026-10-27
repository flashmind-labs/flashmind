"""Run the Flashmind CLI inside Harbor task containers."""

import hashlib
import json
import os
import re
import shlex
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

PROVIDER_KEYS = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": None,
}
REPO = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO / "target/x86_64-unknown-linux-musl/debug/flsh"
REMOTE_BINARY = "/installed-agent/flsh"
REMOTE_HOME = "/tmp/flashmind-harness"


def model_parts(model: str) -> tuple[str, str]:
    """Accept Flashmind's provider:model and Harbor's provider/model forms."""
    match = re.fullmatch(r"([^:/]+)[:/](.+)", model)
    if not match or match[1] not in PROVIDER_KEYS:
        raise ValueError("Use a supported provider:model or provider/model selector")
    return match[1], match[2]


def event_usage(event: dict) -> dict | None:
    while event.get("type") == "spawned_event":
        event = event["data"]["event"]
    return event["data"] if event.get("type") == "usage" else None


class FlashmindAgent(BaseInstalledAgent):
    def __init__(
        self,
        *args,
        binary_path: str = str(DEFAULT_BINARY),
        reasoning: str = "medium",
        tools: str | None = None,
        **kwargs,
    ):
        if reasoning not in {"off", "low", "medium", "high"}:
            raise ValueError("reasoning must be off, low, medium, or high")
        if tools is not None and not isinstance(tools, str):
            raise TypeError("tools must be a comma-separated string")
        self.tool_names = (
            list(dict.fromkeys(name.strip() for name in tools.split(",")))
            if tools is not None
            else None
        )
        if self.tool_names is not None and any(
            re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None for name in self.tool_names
        ):
            raise ValueError(
                "tools must contain nonempty tool names separated by commas"
            )
        self.binary_path = Path(binary_path).expanduser().resolve()
        if not self.binary_path.is_file():
            raise FileNotFoundError("Build flsh first or pass --ak binary_path=...")
        with self.binary_path.open("rb") as binary:
            self.binary_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
        self.reasoning = reasoning
        super().__init__(*args, **kwargs)
        if not self.model_name:
            raise ValueError("Pass --model with an explicit provider and model")

    @staticmethod
    def name() -> str:
        return "flashmind"

    def version(self) -> str:
        return self.binary_sha256

    def _parse_model_name(self, model_name: str) -> tuple[str, str]:
        return model_parts(model_name)

    async def install(self, environment: BaseEnvironment) -> None:
        await environment.upload_file(self.binary_path, REMOTE_BINARY)
        await self.exec_as_root(environment, command=f"chmod 755 {REMOTE_BINARY}")
        await self.exec_as_agent(environment, command=f"{REMOTE_BINARY} --help")

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        provider, model = model_parts(self.model_name)
        env = {"FLASHMIND_HOME": REMOTE_HOME}
        endpoint = {"openai": "OPENAI_BASE_URL", "ollama": "OLLAMA_URL"}.get(provider)
        for name in (PROVIDER_KEYS[provider], endpoint):
            if name:
                value = self._extra_env.get(name, os.environ.get(name))
                if value:
                    env[name] = value
                elif name == PROVIDER_KEYS[provider]:
                    raise ValueError(f"Set {name} before starting the benchmark")

        await self.exec_as_agent(environment, command=f"mkdir -p {REMOTE_HOME}")
        await self._upload_config_text(
            environment,
            content=f'reasoning = "{self.reasoning}"\n',
            remote_path=f"{REMOTE_HOME}/config.toml",
            filename="config.toml",
        )
        logs = shlex.quote(str(self.environment_logs_dir))
        arguments = [
            REMOTE_BINARY,
            "--model",
            f"{provider}:{model}",
            "--dangerously-skip-permissions",
            "--json",
        ]
        if self.tool_names is not None:
            arguments.extend(["--tools", ",".join(self.tool_names)])
        arguments.extend(["--prompt", instruction])
        command = shlex.join(arguments)
        await self.exec_as_agent(
            environment,
            command=f"{command} > {logs}/events.jsonl 2> {logs}/stderr.txt",
            env=env,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        # Harbor calls this hook only while the context is still empty.
        context.metadata = {
            "binary_sha256": self.binary_sha256,
            "reasoning": self.reasoning,
            "tools": self.tool_names,
        }
        events = self.logs_dir / "events.jsonl"
        if events.exists():
            totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cache_read_tokens": 0,
            }
            seen_usage = False
            with events.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # A timeout can leave the final event only partly written.
                        continue
                    usage = event_usage(event)
                    if usage is not None:
                        seen_usage = True
                        for key in totals:
                            totals[key] += usage.get(key, 0)
            if seen_usage:
                context.n_input_tokens = totals["prompt_tokens"]
                context.n_output_tokens = totals["completion_tokens"]
                context.n_cache_tokens = totals["cache_read_tokens"]

        stderr = self.logs_dir / "stderr.txt"
        if stderr.exists():
            match = re.search(
                r"^Cost: \$(\d+(?:\.\d+)?)$", stderr.read_text(), re.MULTILINE
            )
            if match:
                context.cost_usd = float(match[1])
