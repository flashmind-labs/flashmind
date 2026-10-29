# openrouter

Call any OpenRouter model from a shell: text, images, video, speech and
transcription. Built for scripts and coding agents, so it never prompts.

```sh
cargo install flashmind-openrouter-cli
```

Put your key in `~/.openrouter-cli` and `chmod 600` it, or set
`OPENROUTER_API_KEY` if the file does not exist:

```toml
openrouter_key = "YOUR_OPENROUTER_KEY"
```

## Usage

The prompt is the last argument, or stdin when it is omitted.

```sh
openrouter text -m google/gemini-2.5-flash "Summarize this in one line" -a notes.md
git diff | openrouter text -m openai/gpt-5 -s "Review this diff" --reasoning high
openrouter image -m black-forest-labs/flux-3-image --aspect-ratio 16:9 -o cat.png "a cat"
openrouter video -m google/veo-3.1 --duration 8 --first-frame cat.png "the cat jumps"
openrouter speech -m hexgrad/kokoro-82m --voice af_heart -o hello.mp3 "Hello"
openrouter transcribe -m google/gemini-3.5-transcribe hello.mp3
openrouter models image
```

`text` and `transcribe` print text to stdout. `image`, `video` and `speech`
save files and print one path per line. Without `-o`, files go in the current
directory with a generated name; `-o -` writes the bytes to stdout. Progress
goes to stderr. Errors exit nonzero.

For timestamps and speaker labels, request the full JSON response:

```sh
openrouter transcribe -m microsoft/mai-transcribe-2 audio.mp3 --json \
  --timestamps segment,word \
  --provider-options '{"azure":{"diarization":{"enabled":true}}}'
```

`--json` requests `verbose_json` and preserves every response field, including
segments, words, speaker labels and usage. `--timestamps` and `--provider-options`
also enable JSON output. Diarization options use the serving provider's endpoint
slug and parameter names; support depends on the model and provider. See the
[OpenRouter transcription guide](https://openrouter.ai/docs/guides/overview/multimodal/stt).

`-a` attaches images, PDFs, audio, video or text files to a `text` request.
`models` takes `text`, `image`, `video`, `speech` or `transcription`. Run
`openrouter <command> --help` for every option.
