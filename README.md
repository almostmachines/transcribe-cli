# openrouter-transcribe-cli

A polished, dependency-light command-line transcription tool for hosted speech-to-text models through [OpenRouter](https://openrouter.ai/). The installed command is **`transcribe-cli`**.

It defaults to [`openai/gpt-transcribe`](https://openrouter.ai/openai/gpt-transcribe), while `--model` makes it easy to switch providers without changing the workflow.

## Highlights

- Audio **and video** input
- Plain text, JSON, SRT, and WebVTT output
- Automatic media normalization and long-recording chunking with `ffmpeg`
- Concurrent chunk uploads, retries, and rate-limit handling
- Language and vocabulary/context hints
- Word timestamps where the provider supports them
- Multiple files, stdin, safe atomic output, and dry-run plans
- Cost/usage reporting when OpenRouter returns it
- No Python runtime dependencies

## Install

Python 3.11 or newer is required. `ffmpeg`/`ffprobe` are only needed for videos, unknown formats, and chunking.

```bash
uv tool install .
```

For development:

```bash
uv venv
uv pip install -e .
```

## Authentication

The CLI first looks for a dedicated variable and then the conventional OpenRouter variable:

```bash
export OPENROUTER_TRANSCRIPTION_KEY=sk-or-v1-...
# or: export OPENROUTER_API_KEY=sk-or-v1-...
```

The key is only sent in the HTTPS `Authorization` header. There is intentionally no command-line key flag, so secrets do not leak into shell history or process listings.

## Usage

```bash
# Print a transcript to stdout
transcribe-cli voice-note.m4a

# Write text to a file
transcribe-cli meeting.mp4 -o meeting.txt

# Subtitles; the format is inferred from the suffix
transcribe-cli interview.mov -o interview.srt
transcribe-cli interview.mov -o interview.vtt

# Rich metadata and word timestamps
transcribe-cli call.mp3 -o call.json --word-timestamps

# Language and vocabulary hints
transcribe-cli recording.ogg -l en \
  --prompt 'Names and terms: Pi, OpenRouter.'

# Process chunks in parallel
transcribe-cli long-recording.mkv -o transcript.txt --chunk-seconds 600 --jobs 3

# Batch mode writes one output per input
transcribe-cli *.m4a --output-dir transcripts --format txt

# Pipe media in
cat voice-note.opus | transcribe-cli -

# Inspect the plan without spending anything
transcribe-cli meeting.mp4 -o meeting.srt --dry-run

# Discover current STT choices on OpenRouter
transcribe-cli --list-models
```

Run `transcribe-cli --help` for every option.

## Output behavior

- One input with no `--output`: stdout.
- One input with `--output`: that path; format inferred from `.txt`, `.json`, `.srt`, or `.vtt`.
- Multiple inputs: output alongside each source, or under `--output-dir`.
- Existing files are protected unless `--force` is given.
- Files are written atomically, so a failed request does not leave a partial transcript.

JSON output contains source/model metadata, detected language when available, full text, offset-corrected segments and words, per-request generation IDs, chunk metadata, and aggregated OpenRouter usage.

## Long recordings

The default chunk size is 10 minutes. Long or oversized inputs are converted to 16 kHz mono Opus before upload, keeping requests comfortably below provider upload limits and avoiding long upstream timeouts. Use `--chunk-seconds 0` to disable splitting.

Chunks are sequential by default. `--jobs N` can reduce wall-clock time, but may consume provider rate limits more quickly.

## Model portability

Choose any model supported by OpenRouter's speech-to-text endpoint:

```bash
transcribe-cli audio.wav --model openai/gpt-transcribe
transcribe-cli audio.wav --model deepgram/nova-3
transcribe-cli audio.wav --model mistralai/voxtral-mini-transcribe
```

Some providers do not implement verbose or word-level timestamps. For those models, the CLI automatically falls back to chunk-level timestamps instead of failing the whole job.

Set a persistent default with:

```bash
export TRANSCRIBE_MODEL=openai/gpt-transcribe
```

Other useful environment variables are `TRANSCRIBE_CHUNK_SECONDS` and `OPENROUTER_TRANSCRIPTION_URL`.
