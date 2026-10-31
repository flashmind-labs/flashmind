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

Use `/model` to switch models, `/status` to inspect the session, and
`/compact` to compress conversation context. `/new` starts a session,
`/resume` switches to an earlier one, and `/export` writes a conversation to
Markdown. Use `/skills` and `/agents` to list available skills and agents.

## Text and media

Replace `<model>` and `<voice>` in the examples with values supported by your
provider. Use `flsh <command> --help` for the available options. Model support
for formats, sizes, references and other options varies by provider.

### Generate text

```sh
flsh text -m 'openrouter:<model>' "Write a short poem"
flsh text -m 'openrouter:<model>' --attach notes.txt "Summarize this file"
cat notes.txt | flsh text -m 'openrouter:<model>'
```

Omit the prompt to read it from standard input. Use `--attach` repeatedly to
attach files, `--system` to set a system prompt, and `--reasoning` to choose an
effort level. For example, `cat notes.txt | flsh text -m 'openrouter:<model>'`
uses the file contents as the prompt.

### Generate an image

```sh
flsh image gen -m 'openrouter:<model>' -o fox.png "A red fox in a snowy forest"
flsh image gen -m 'openrouter:<model>' --aspect-ratio 16:9 --size 1K -o landscape.png "A mountain lake at sunrise"
```

`-o` sets the output path. Use `--aspect-ratio`, `--size`, `--quality` and
`--format` when the model supports them. Use `-n` for multiple images and
repeat `--reference` to supply reference images as paths or URLs. Extra output
files receive numbered names, such as `fox-2.png`. `flsh image` also accepts
the generation options directly, without `gen`.

### Edit an image

Choose an edit-capable model and provide the image and instructions:

```sh
flsh image edit -m 'openrouter:<model>' --image fox.png -o fox-hat.png "Add a blue hat to the fox"
flsh image edit -m 'openrouter:<model>' --image fox.png --image hat.png -o fox-hat.png "Put the hat from the second image on the fox"
```

Repeat `--image` for multiple inputs, with the base image first. Inputs can be
local paths or URLs. `--resolution`, `--aspect-ratio`, `--quality` and
`--format` depend on the model. OpenAI image editing also supports `--mask`
with a PNG: transparent areas mark the parts to edit.

### Generate a video

```sh
flsh video -m 'openrouter:<model>' -o waves.mp4 "Waves rolling onto a beach at dusk"
flsh video -m 'openrouter:<model>' --first-frame fox.png --duration 5 --resolution 720p -o fox.mp4 "The fox walks through the snow"
```

Use `--duration` for seconds, `--resolution` and `--aspect-ratio` for output
settings, and `--audio` to request generated audio. `--first-frame` and
`--last-frame` accept an image path or URL. Repeat `--reference` for image
references. Availability depends on the selected model and provider.

Some video models accept a source clip as a generation reference:

```sh
flsh video -m 'openrouter:<model>' --video-reference source.mp4 -o result.mp4 "Create a watercolor version of this scene"
```

Repeat `--video-reference` for additional paths or URLs when supported. This
sends the clip as a reference to a video generation model; results depend on
the model. `flsh` does not have a separate video editor for operations such as
trimming clips or changing a timeline.

### Speech, transcription and OCR

```sh
flsh voices openrouter
flsh speech -m 'openrouter:<model>' --voice '<voice>' -o speech.mp3 "Hello"
flsh transcribe -m 'openrouter:<model>' --language en audio.mp3
flsh ocr document.pdf
```

`voices` lists a provider's available voices. `speech` accepts `--format` for
the audio format. `transcribe --json` prints the provider's full response;
`--timestamps segment,word` requests timings when supported. OCR uses Mistral
and accepts a PDF, image or public URL. Add `--json` for the full OCR response,
or `--image` if a URL points to an image without a recognizable extension.

Image, video and speech commands choose a new output file when `-o` is omitted.
Use `--overwrite` to replace an existing file or `-o -` to write to standard
output. When a text, image, video or speech prompt is omitted, it is read from
standard input. Use `-e 'key=JSON_VALUE'` for a provider-specific request
field; repeat the option for multiple fields.

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

### Rig a character with Meshy

Generate a textured humanoid and add a skeleton in the same command:

```sh
flsh 3d -m meshy:latest --texture --rig -o rigged.glb "A humanoid in a T-pose, with arms and legs separated"
```

`--texture` generates the textures required by Meshy's rigging API. `--rig`
then submits a separate rigging task and saves the rigged model to the path
given by `-o`. With text input, Meshy first generates a preview and then runs a
refine task for the texture.

To add a skeleton to an existing textured GLB instead, use `--rig-model`:

```sh
flsh 3d -m meshy:latest --rig --rig-model character.glb -o rigged.glb
```

The input can be a local GLB up to 100 MiB or a publicly accessible HTTP(S)
URL. It must be a humanoid with clearly separated limbs, facing the +Z axis.
An A-pose or T-pose works best. The output is a new GLB with the skeleton and
skinning; `flsh` will not overwrite an existing output file.

Meshy's rigging API accepts models with at most 380,000 triangles. Add
`--remesh` for a larger model, and use `--height` to set the character's
approximate height in meters:

```sh
flsh 3d -m meshy:latest --rig --rig-model character.glb --remesh --height 1.8 -o rigged.glb
```

`--remesh` submits a separate task targeting 300,000 triangles; the actual
count can differ. Without `--height`, Meshy uses 1.7 meters. Generation,
texturing, remeshing and rigging are separate tasks that can consume provider
credits. `flsh` prints each task ID as it is submitted; an interrupted task
may continue running on Meshy.

## Shell completion

Run `flsh completions install` to configure your login shell. Start a new shell
or source its startup file to activate completion. To generate a script
instead, use `flsh completions bash`, `zsh`, or `fish`.

## Sessions and maintenance

```sh
flsh resume --all
flsh resume '<session-id>'
flsh import pi
flsh logs yesterday -n 100
flsh logs --follow
flsh clean --days 30 --dry-run
flsh update
```

`resume --all` includes sessions from other directories; pass a session ID to
open one directly. `import pi` imports pi sessions and accepts an optional
path. `logs` shows agent logs, with `--follow` for new output. `clean` removes
expired logs and sessions; use `--dry-run` to preview removal. `update`
installs the latest version from the Git repository.
