# Agent task benchmarks

The [Harbor](https://docs.harborframework.com/agents/custom-agents) adapter runs
the real `flsh` CLI inside each task container, using its default prompt, tools,
subagents, and compaction. Harbor runs the task's verifier afterward.

Run from the repository root on Linux with Docker, Python 3.12+, and uv:

```sh
uv venv target/harness-bench/venv
uv pip install --python target/harness-bench/venv/bin/python -r benchmarks/requirements.txt
rustup target add x86_64-unknown-linux-musl
# Install musl-tools with your system package manager if musl-gcc is missing.
cargo build -p flashmind-cli --target x86_64-unknown-linux-musl

# Set OPENROUTER_API_KEY in the environment first.
PYTHONPATH=. target/harness-bench/venv/bin/harbor run \
  --dataset terminal-bench@2.0 \
  --agent benchmarks.harbor_agent:FlashmindAgent \
  --model openrouter/example/model \
  --ak reasoning=medium \
  --include-task-name fix-git \
  --include-task-name log-summary-date-ranges \
  --include-task-name large-scale-text-editing \
  --include-task-name sqlite-db-truncate \
  --include-task-name cancel-async-tasks \
  --n-concurrent 2 --max-retries 0 \
  --jobs-dir target/harness-bench/jobs
```

This five-task subset is a smoke run, not a Terminal-Bench leaderboard score.
Remove the task filters for all 89 tasks. Keep the model, reasoning level,
dataset version, task selection, and attempt count fixed when comparing harnesses.
Other Harbor datasets, including SWE-bench, use the same adapter.

Compare Flashmind, Pi, and OpenCode with one model and task selection:

```sh
target/harness-bench/venv/bin/python benchmarks/compare.py \
  --model openrouter/example/model
```

The comparison pins Pi and OpenCode versions and sets high reasoning for all
three harnesses. It keeps their default prompts and tools, gives each two
concurrent trials, and uses the task's standard time and resource limits.
Other model settings use each harness's defaults.
Use `--dry-run` to inspect the configuration, `--agents pi opencode` to run only
the baselines, or `--all-tasks --attempts 3` for a broader repeated comparison.
Compare verifier grades and execution time per task. Treat provider errors
separately, and report missing cost as unknown.
Check effective reasoning in the saved sessions: a harness may promote an
unsupported level. Pi promotes medium to high for some models.

To compare Flashmind tool sets, keep the same binary and model and run once
with `--flashmind-tools all` and once with the core set:

```sh
target/harness-bench/venv/bin/python benchmarks/compare.py \
  --model openrouter/example/model --agents flashmind \
  --flashmind-tools read,write,edit,find,exec,process
```

Only the exposed tools and their generated tool instructions change. The base
prompt stays the same. The adapter also accepts `--ak tools=read,write,edit,find,exec,process`.

The adapter accepts `--ak binary_path=/path/to/flsh` for another Linux binary.
Use a static binary when task images have older libc versions. Models accept
`provider:model` or `provider/model`; supported providers are OpenRouter,
Anthropic, OpenAI, and Ollama. Pass credentials through the matching environment
variable. OpenAI and Ollama also accept `OPENAI_BASE_URL` and `OLLAMA_URL`.
Ollama needs an endpoint reachable from inside the container.

Each task gets fresh Flashmind settings and state, with medium reasoning by
default. Host settings, instructions, skills, and session history are not copied.
Shell approval is disabled inside the task container. Model calls incur the
provider's normal charges.

Harbor's job directory contains verifier rewards and execution times. Each
trial's `agent/events.jsonl` and `agent/stderr.txt` contain the CLI trace and
usage summary. Token counts include subagent events; cost is populated when the
CLI reports it. Missing usage stays unknown. Results record the binary SHA-256
and reasoning level. Keep raw traces local; they may contain task data.

```sh
target/harness-bench/venv/bin/python -m unittest discover -s benchmarks -p 'test_*.py'
```
