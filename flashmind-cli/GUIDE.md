# flsh guide

`flsh` is a terminal AI assistant and a command-line interface for generating
text and media. Provider availability and supported options depend on the
model you choose.

## Install and set up

```sh
cargo install flashmind-cli
flsh setup
```

Setup lets you choose a provider and model. You can also supply credentials
through environment variables, such as `OPENROUTER_API_KEY`, `MESHY_API_KEY`,
or `MISTRAL_API_KEY`. Keep keys out of source control.

Run `flsh models` to browse models, or `flsh models openrouter` to filter by
provider. Select a model for a command with `-m provider:model`.

## Chat and coding

Run `flsh` in a project directory for interactive chat, or send one prompt:

```sh
flsh
flsh -p "Explain this project"
flsh -m 'openrouter:<model>' -p "Find the tests for this module"
flsh resume
```

The agent can use tools, including file editing and shell commands. Shell
commands run without approval prompts. Use `--tools` with a comma-separated list
to restrict the tools available. In a session, `/undo` removes the last exchange
from the conversation and `/retry` sends it again; neither undoes file changes.
Use `flsh --help` and the interactive `/help` for more options.

## Text and media

Replace `<model>` and `<voice>` with values available from your provider.
`flsh <command> --help` shows each command's options.

```sh
flsh text -m 'openrouter:<model>' "Write a short poem"
flsh image gen -m 'openrouter:<model>' -o image.png "A fox"
flsh image edit -m 'openrouter:<model>' --image image.png -o edited.png "Add a hat"
flsh video -m 'openrouter:<model>' -o video.mp4 "Waves at dusk"
flsh voices openrouter
flsh speech -m 'openrouter:<model>' --voice '<voice>' -o speech.mp3 "Hello"
flsh transcribe -m 'openrouter:<model>' audio.mp3
flsh ocr document.pdf
```

OCR uses Mistral. Image editing, speech and other media commands need a model
that supports the requested operation. Use `-e 'key=JSON_VALUE'` to pass a
provider-specific request field; repeat it for more fields.

## Generate 3D models

Set `MESHY_API_KEY` or `TRIPO_API_KEY`, then generate a GLB file from a prompt
or reference image:

```sh
flsh 3d -m meshy:latest -o chair.glb "A wooden chair"
flsh 3d -m meshy:latest --image front.png --texture -o character.glb
flsh 3d -m meshy:latest --image front.png --image side.png --image back.png --texture -o character-views.glb
```

Use a clear image of one subject on a plain background. For multiple views,
use matching images of the same subject and pose. Up to four images are
accepted, as local paths or HTTP(S) URLs. Tripo expects front, left, back, then
right. `--texture` adds textures; without it the result is untextured. For
Meshy text-to-3D, texturing creates a separate refine task. Generation consumes
provider credits, and a submitted task can continue after you interrupt flsh.

Rig a Meshy humanoid after generation, or rig an existing textured GLB:

```sh
flsh 3d -m meshy:latest --texture --rig -o rigged.glb "Humanoid in a T-pose"
flsh 3d -m meshy:latest --rig --rig-model character.glb --remesh --height 1.8 -o rigged.glb
```

Meshy's rigging API needs a textured humanoid with at most 380,000 triangles.
`--remesh` runs a separate task targeting 300,000 triangles before rigging.
Existing input can be a local GLB (up to 100 MiB) or a public HTTP(S) URL.
For URL or file inputs, the character must face the +Z axis. Each task uses
provider credits; the remesh target is approximate.

## Shell completion

Run `flsh completions install` to configure your login shell. Start a new shell
or source its startup file to activate completion. To generate a script
instead, use `flsh completions bash`, `zsh`, or `fish`.
