---
name: openrouter
description: Call other AI models through OpenRouter to generate text, images, video or speech, or to transcribe audio.
---

# OpenRouter models

Use `openrouter` when you need another model's answer, or output you cannot
produce yourself: images, video, speech or a transcription. The user installs
it with:

```sh
cargo install --git https://github.com/flashmind-labs/flashmind openrouter-cli
```

The key lives in `~/.openrouter-cli` as `openrouter_key = "..."`, or in
`OPENROUTER_API_KEY`.

Every command needs `-m <model id>`. Model IDs change often, so list current
ones instead of guessing:

```sh
openrouter models          # text models
openrouter models image
openrouter models video
openrouter models speech
openrouter models transcription
```

Pass the prompt as the last argument, or on stdin when it is omitted.

```sh
openrouter text -m <model> "Question"
openrouter text -m <model> -s "System prompt" -a screenshot.png -a spec.pdf "What is wrong here?"
openrouter image -m <model> --aspect-ratio 1:1 -o logo.png "A flat fox logo"
openrouter video -m <model> --duration 5 --resolution 720p -o clip.mp4 "Waves at dusk"
openrouter speech -m <model> --voice <voice> -o line.mp3 "Welcome back"
openrouter transcribe -m <model> meeting.mp3
```

`text` and `transcribe` print text to stdout. `image`, `video` and `speech`
print the path of each saved file, one per line; read those paths rather than
assuming the name. Video jobs can take several minutes; progress goes to
stderr. A nonzero exit means failure, with the reason on stderr.

Each call costs money. Do not loop on retries, and keep secrets out of
prompts and attachments.
