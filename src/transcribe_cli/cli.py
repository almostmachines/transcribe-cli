"""Command-line interface for hosted transcription through OpenRouter."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import math
import mimetypes
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__

DEFAULT_MODEL = "openai/gpt-transcribe"
DEFAULT_API_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
MODELS_URL = "https://openrouter.ai/api/v1/models?input_modalities=audio&output_modalities=transcription"
API_KEY_NAMES = ("OPENROUTER_TRANSCRIPTION_KEY", "OPENROUTER_API_KEY")
OUTPUT_FORMATS = ("txt", "json", "srt", "vtt")
DIRECT_FORMATS = {
    ".aac": "aac",
    ".flac": "flac",
    ".m4a": "m4a",
    ".mp3": "mp3",
    ".oga": "ogg",
    ".ogg": "ogg",
    ".opus": "ogg",
    ".wav": "wav",
    ".webm": "webm",
}
DIRECT_SIZE_LIMIT = 20 * 1024 * 1024


class CliError(Exception):
    """A concise, user-facing error."""


class ApiError(CliError):
    """An OpenRouter API error."""

    def __init__(self, status: int | None, message: str):
        self.status = status
        super().__init__(message)


@dataclass(frozen=True)
class Chunk:
    path: Path
    index: int
    total: int
    start: float
    duration: float | None


@dataclass
class ChunkResult:
    chunk: Chunk
    response: dict[str, Any]
    generation_id: str | None


@dataclass
class Transcript:
    source: str
    model: str
    language: str | None
    duration: float | None
    text: str
    segments: list[dict[str, Any]]
    words: list[dict[str, Any]]
    usage: dict[str, float | int]
    generation_ids: list[str]
    chunks: list[dict[str, Any]]


class Reporter:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self._lock = threading.Lock()

    def __call__(self, message: str) -> None:
        if self.quiet:
            return
        with self._lock:
            print(message, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio or video with hosted models through OpenRouter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", metavar="FILE", help="Audio/video file; use - for stdin.")
    parser.add_argument("-o", "--output", type=Path, help="Output path for one input; - means stdout.")
    parser.add_argument("-d", "--output-dir", type=Path, help="Output directory, especially for batches.")
    parser.add_argument("-f", "--format", choices=OUTPUT_FORMATS, help="Output format; inferred from -o when possible.")
    parser.add_argument("-m", "--model", default=os.getenv("TRANSCRIBE_MODEL", DEFAULT_MODEL), help="OpenRouter model slug.")
    parser.add_argument("-l", "--language", help="ISO-639-1 language hint, e.g. en, fr, or ja.")
    parser.add_argument("--prompt", help="Vocabulary/context hint (some providers may ignore it).")
    parser.add_argument("--temperature", type=float, help="Sampling temperature from 0 to 1.")
    parser.add_argument("--word-timestamps", action="store_true", help="Request word timestamps in JSON output.")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=int(os.getenv("TRANSCRIBE_CHUNK_SECONDS", "600")),
        help="Split long media into chunks; 0 disables splitting.",
    )
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Chunks to submit concurrently.")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout per attempt in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries for rate limits and transient failures.")
    parser.add_argument("--api-url", default=os.getenv("OPENROUTER_TRANSCRIPTION_URL", DEFAULT_API_URL), help=argparse.SUPPRESS)
    parser.add_argument("--api-key-env", default="OPENROUTER_TRANSCRIPTION_KEY", help="Environment variable containing the API key.")
    parser.add_argument("--force", action="store_true", help="Replace existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without converting media or calling the API.")
    parser.add_argument("--list-models", action="store_true", help="List OpenRouter speech-to-text models and exit.")
    parser.add_argument("--models-json", action="store_true", help="With --list-models, emit machine-readable JSON.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress status and usage messages.")
    parser.add_argument("--version", action="version", version=f"transcribe {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reporter = Reporter(args.quiet)
    try:
        validate_args(args)
        if args.list_models:
            return list_models(args)
        return run(args, reporter)
    except CliError as exc:
        parser.exit(2, f"transcribe: error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "transcribe: interrupted\n")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.list_models:
        return
    if not args.inputs:
        raise CliError("at least one input file is required (or use --list-models)")
    if len(args.inputs) > 1 and args.output is not None:
        raise CliError("--output can only be used with one input; use --output-dir for batches")
    if args.output is not None and args.output_dir is not None:
        raise CliError("--output and --output-dir cannot be used together")
    if args.inputs.count("-") > 1 or ("-" in args.inputs and len(args.inputs) > 1):
        raise CliError("stdin (-) must be the only input")
    if args.chunk_seconds < 0:
        raise CliError("--chunk-seconds must be zero or greater")
    if args.jobs < 1:
        raise CliError("--jobs must be at least 1")
    if args.timeout <= 0:
        raise CliError("--timeout must be greater than zero")
    if args.retries < 0:
        raise CliError("--retries must be zero or greater")
    if args.temperature is not None and not 0 <= args.temperature <= 1:
        raise CliError("--temperature must be between 0 and 1")
    if args.output is not None and str(args.output) == "-" and args.output_dir is not None:
        raise CliError("stdout output cannot be combined with --output-dir")


def run(args: argparse.Namespace, reporter: Reporter) -> int:
    output_format = infer_output_format(args.output, args.format)
    output_paths = resolve_output_paths(args, output_format)
    preflight_outputs(output_paths, args.force)

    api_key = None if args.dry_run else resolve_api_key(args.api_key_env)

    with tempfile.TemporaryDirectory(prefix="transcribe-") as temporary:
        temp_root = Path(temporary)
        sources = materialize_inputs(args.inputs, temp_root)
        for position, ((label, source, force_convert), output_path) in enumerate(
            zip(sources, output_paths), start=1
        ):
            prefix = f"[{position}/{len(sources)}] " if len(sources) > 1 else ""
            duration = probe_duration(source)
            if args.dry_run:
                print_plan(label, source, duration, output_path, output_format, args, reporter, prefix)
                continue

            reporter(f"{prefix}Preparing {label} ({format_duration(duration)})...")
            work_dir = temp_root / f"media-{position:04d}"
            work_dir.mkdir()
            chunks = prepare_chunks(
                source,
                work_dir,
                duration=duration,
                chunk_seconds=args.chunk_seconds,
                force_convert=force_convert,
            )
            transcript = transcribe_chunks(
                chunks,
                source_label=label,
                source_duration=duration,
                model=args.model,
                api_url=args.api_url,
                api_key=api_key or "",
                language=args.language,
                prompt=args.prompt,
                temperature=args.temperature,
                word_timestamps=args.word_timestamps,
                output_format=output_format,
                jobs=args.jobs,
                retries=args.retries,
                timeout=args.timeout,
                reporter=reporter,
                status_prefix=prefix,
            )
            content = render_transcript(transcript, output_format)
            write_output(output_path, content, force=args.force)
            report_completion(transcript, output_path, output_format, reporter, prefix)
    return 0


def resolve_api_key(preferred_name: str) -> str:
    names = tuple(dict.fromkeys((preferred_name, *API_KEY_NAMES)))
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined = " or ".join(names)
    raise CliError(f"API key not found; export {joined}")


def infer_output_format(output: Path | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if output is not None and str(output) != "-":
        suffix = output.suffix.lower().lstrip(".")
        if suffix in OUTPUT_FORMATS:
            return suffix
    return "txt"


def resolve_output_paths(args: argparse.Namespace, output_format: str) -> list[Path | None]:
    if args.output is not None:
        return [None if str(args.output) == "-" else args.output]
    if len(args.inputs) == 1 and args.output_dir is None:
        return [None]

    outputs: list[Path | None] = []
    for raw in args.inputs:
        if raw == "-":
            outputs.append(None)
            continue
        source = Path(raw)
        directory = args.output_dir if args.output_dir is not None else source.parent
        outputs.append(directory / f"{source.stem}.{output_format}")
    return outputs


def preflight_outputs(outputs: Sequence[Path | None], force: bool) -> None:
    concrete = [path.resolve() for path in outputs if path is not None]
    if len(concrete) != len(set(concrete)):
        raise CliError("multiple inputs resolve to the same output path")
    if not force:
        existing = next((path for path in outputs if path is not None and path.exists()), None)
        if existing is not None:
            raise CliError(f"output already exists: {existing} (use --force to replace it)")


def materialize_inputs(inputs: Sequence[str], temp_root: Path) -> list[tuple[str, Path, bool]]:
    result: list[tuple[str, Path, bool]] = []
    for index, raw in enumerate(inputs):
        if raw == "-":
            target = temp_root / f"stdin-{index:04d}.media"
            with target.open("wb") as destination:
                shutil.copyfileobj(sys.stdin.buffer, destination)
            if target.stat().st_size == 0:
                raise CliError("stdin contained no data")
            result.append(("stdin", target, True))
            continue
        path = Path(raw).expanduser()
        if not path.exists():
            raise CliError(f"input not found: {path}")
        if not path.is_file():
            raise CliError(f"input is not a file: {path}")
        result.append((str(path), path, False))
    return result


def probe_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    try:
        value = float(completed.stdout.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def prepare_chunks(
    source: Path,
    work_dir: Path,
    *,
    duration: float | None,
    chunk_seconds: int,
    force_convert: bool,
) -> list[Chunk]:
    direct_format = DIRECT_FORMATS.get(source.suffix.lower())
    should_split = bool(
        chunk_seconds
        and (
            (duration is not None and duration > chunk_seconds)
            or source.stat().st_size > DIRECT_SIZE_LIMIT
        )
    )
    if direct_format and not force_convert and not should_split:
        return [Chunk(source, 1, 1, 0.0, duration)]

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise CliError("ffmpeg is required for video, unknown formats, and long recordings")

    if should_split:
        pattern = work_dir / "chunk-%05d.ogg"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            str(pattern),
        ]
    else:
        target = work_dir / "chunk-00000.ogg"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            str(target),
        ]

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown ffmpeg error"
        raise CliError(f"could not decode media: {detail}")

    paths = sorted(work_dir.glob("chunk-*.ogg"))
    if not paths:
        raise CliError("ffmpeg produced no audio chunks; the input may not contain an audio stream")

    durations = [probe_duration(path) for path in paths]
    chunks: list[Chunk] = []
    offset = 0.0
    for index, (path, part_duration) in enumerate(zip(paths, durations), start=1):
        chunks.append(Chunk(path, index, len(paths), offset, part_duration))
        if part_duration is not None:
            offset += part_duration
        elif chunk_seconds:
            offset += float(chunk_seconds)
    return chunks


def transcribe_chunks(
    chunks: Sequence[Chunk],
    *,
    source_label: str,
    source_duration: float | None,
    model: str,
    api_url: str,
    api_key: str,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
    word_timestamps: bool,
    output_format: str,
    jobs: int,
    retries: int,
    timeout: float,
    reporter: Reporter,
    status_prefix: str,
) -> Transcript:
    want_verbose = output_format in {"json", "srt", "vtt"} or word_timestamps

    def submit(chunk: Chunk) -> ChunkResult:
        span = format_chunk_span(chunk)
        reporter(f"{status_prefix}[chunk {chunk.index}/{chunk.total} {span}] Transcribing...")
        try:
            response, generation_id = request_transcription(
                chunk.path,
                api_url=api_url,
                api_key=api_key,
                model=model,
                language=language,
                prompt=prompt,
                temperature=temperature,
                verbose=want_verbose,
                word_timestamps=word_timestamps,
                timeout=timeout,
                retries=retries,
            )
        except ApiError as exc:
            if want_verbose and exc.status == 400:
                reporter(
                    f"{status_prefix}[chunk {chunk.index}/{chunk.total}] "
                    "Provider rejected timestamp output; falling back to chunk-level timestamps."
                )
                response, generation_id = request_transcription(
                    chunk.path,
                    api_url=api_url,
                    api_key=api_key,
                    model=model,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    verbose=False,
                    word_timestamps=False,
                    timeout=timeout,
                    retries=retries,
                )
            else:
                raise
        return ChunkResult(chunk, response, generation_id)

    if jobs == 1 or len(chunks) == 1:
        results = [submit(chunk) for chunk in chunks]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(jobs, len(chunks))) as pool:
            futures = {pool.submit(submit, chunk): chunk for chunk in chunks}
            unordered: list[ChunkResult] = []
            try:
                for future in concurrent.futures.as_completed(futures):
                    unordered.append(future.result())
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        results = sorted(unordered, key=lambda item: item.chunk.index)

    return combine_results(
        results,
        source_label=source_label,
        source_duration=source_duration,
        model=model,
        requested_language=language,
    )


def request_transcription(
    audio_path: Path,
    *,
    api_url: str,
    api_key: str,
    model: str,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
    verbose: bool,
    word_timestamps: bool,
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any], str | None]:
    fields: list[tuple[str, str]] = [("model", model)]
    fields.append(("response_format", "verbose_json" if verbose else "json"))
    if language:
        fields.append(("language", language))
    if prompt:
        fields.append(("prompt", prompt))
    if temperature is not None:
        fields.append(("temperature", str(temperature)))
    if verbose and word_timestamps:
        fields.append(("timestamp_granularities[]", "word"))

    body, content_type = encode_multipart(fields, audio_path)
    request = urllib.request.Request(
        api_url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
            "Accept": "application/json",
            "User-Agent": f"openrouter-transcribe-cli/{__version__}",
            "X-OpenRouter-Title": "openrouter-transcribe-cli",
        },
    )

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
                generation_id = response.headers.get("X-Generation-Id")
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                raise ApiError(None, "API returned a malformed transcription response")
            return payload, generation_id
        except urllib.error.HTTPError as exc:
            message = read_api_error(exc)
            if attempt < retries and exc.code in {408, 409, 429, 500, 502, 503, 504}:
                sleep_before_retry(attempt, exc.headers.get("Retry-After"))
                continue
            raise ApiError(exc.code, f"OpenRouter returned HTTP {exc.code}: {message}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries:
                sleep_before_retry(attempt, None)
                continue
            reason = getattr(exc, "reason", exc)
            raise ApiError(None, f"could not reach OpenRouter: {reason}") from exc
        except json.JSONDecodeError as exc:
            raise ApiError(None, "OpenRouter returned invalid JSON") from exc
    raise AssertionError("unreachable")


def encode_multipart(fields: Sequence[tuple[str, str]], audio_path: Path) -> tuple[bytes, str]:
    boundary = f"----transcribe-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    filename = audio_path.name.replace('"', "_")
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            audio_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def read_api_error(error: urllib.error.HTTPError) -> str:
    try:
        raw = error.read(1_000_000).decode("utf-8", errors="replace")
    except OSError:
        return str(error.reason)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip() or str(error.reason)
    detail = payload.get("error", payload) if isinstance(payload, dict) else payload
    if isinstance(detail, dict):
        message = detail.get("message")
        if message:
            return str(message)
    return raw.strip() or str(error.reason)


def sleep_before_retry(attempt: int, retry_after: str | None) -> None:
    if retry_after:
        try:
            delay = min(float(retry_after), 60.0)
        except ValueError:
            delay = 0.0
    else:
        delay = min(2**attempt + random.random(), 30.0)
    time.sleep(max(delay, 0.0))


def combine_results(
    results: Sequence[ChunkResult],
    *,
    source_label: str,
    source_duration: float | None,
    model: str,
    requested_language: str | None,
) -> Transcript:
    text_parts: list[str] = []
    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    usage: dict[str, float | int] = {}
    generation_ids: list[str] = []
    chunks_metadata: list[dict[str, Any]] = []
    detected_language = requested_language

    for result in results:
        response = result.response
        chunk = result.chunk
        text = response.get("text", "").strip()
        if text:
            text_parts.append(text)
        detected_language = detected_language or string_or_none(response.get("language"))
        if result.generation_id:
            generation_ids.append(result.generation_id)

        response_segments = response.get("segments")
        if isinstance(response_segments, list) and response_segments:
            for raw_segment in response_segments:
                if not isinstance(raw_segment, dict):
                    continue
                segment = dict(raw_segment)
                offset_timestamps(segment, chunk.start)
                nested_words = segment.get("words")
                if isinstance(nested_words, list):
                    adjusted = []
                    for raw_word in nested_words:
                        if isinstance(raw_word, dict):
                            word = dict(raw_word)
                            offset_timestamps(word, chunk.start)
                            adjusted.append(word)
                    segment["words"] = adjusted
                segments.append(segment)
        elif text:
            end = chunk.start + (chunk.duration or 0.0)
            segments.append({"start": chunk.start, "end": end, "text": text})

        response_words = response.get("words")
        if isinstance(response_words, list):
            for raw_word in response_words:
                if isinstance(raw_word, dict):
                    word = dict(raw_word)
                    offset_timestamps(word, chunk.start)
                    words.append(word)

        raw_usage = response.get("usage")
        if isinstance(raw_usage, dict):
            accumulate_numeric(usage, raw_usage)
        chunks_metadata.append(
            {
                "index": chunk.index,
                "start": chunk.start,
                "duration": chunk.duration,
                "text": text,
                "usage": raw_usage if isinstance(raw_usage, dict) else None,
                "generation_id": result.generation_id,
            }
        )

    for index, segment in enumerate(segments):
        segment["id"] = index

    duration = source_duration
    if duration is None and results:
        final = results[-1].chunk
        if final.duration is not None:
            duration = final.start + final.duration

    return Transcript(
        source=source_label,
        model=model,
        language=detected_language,
        duration=duration,
        text="\n".join(text_parts),
        segments=segments,
        words=words,
        usage=usage,
        generation_ids=generation_ids,
        chunks=chunks_metadata,
    )


def offset_timestamps(item: dict[str, Any], offset: float) -> None:
    for name in ("start", "end"):
        value = item.get(name)
        if isinstance(value, (int, float)):
            item[name] = value + offset


def accumulate_numeric(target: dict[str, float | int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        target[key] = target.get(key, 0) + value


def string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def render_transcript(transcript: Transcript, output_format: str) -> str:
    if output_format == "txt":
        return transcript.text.rstrip() + ("\n" if transcript.text else "")
    if output_format == "json":
        payload = {
            "source": transcript.source,
            "model": transcript.model,
            "language": transcript.language,
            "duration": transcript.duration,
            "text": transcript.text,
            "segments": transcript.segments,
            "words": transcript.words or None,
            "usage": transcript.usage or None,
            "generation_ids": transcript.generation_ids,
            "chunks": transcript.chunks,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output_format == "srt":
        return render_srt(transcript.segments)
    if output_format == "vtt":
        return render_vtt(transcript.segments)
    raise CliError(f"unsupported output format: {output_format}")


def render_srt(segments: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = numeric_timestamp(segment.get("start"))
        end = numeric_timestamp(segment.get("end"))
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{format_timestamp(start, ',')} --> {format_timestamp(end, ',')}\n"
            f"{text}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(segments: Sequence[dict[str, Any]]) -> str:
    blocks = ["WEBVTT\n"]
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = numeric_timestamp(segment.get("start"))
        end = numeric_timestamp(segment.get("end"))
        blocks.append(
            f"{format_timestamp(start, '.')} --> {format_timestamp(end, '.')}\n{text}\n"
        )
    return "\n".join(blocks) + ("\n" if len(blocks) > 1 else "")


def numeric_timestamp(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def format_timestamp(seconds: float, decimal_marker: str) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds_value, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}{decimal_marker}{milliseconds:03d}"


def write_output(path: Path | None, content: str, *, force: bool) -> None:
    if path is None:
        sys.stdout.write(content)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise CliError(f"output already exists: {path} (use --force to replace it)")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary_path, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()


def report_completion(
    transcript: Transcript,
    output_path: Path | None,
    output_format: str,
    reporter: Reporter,
    prefix: str,
) -> None:
    destination = str(output_path) if output_path is not None else "stdout"
    details = []
    cost = transcript.usage.get("cost")
    seconds = transcript.usage.get("seconds")
    if isinstance(cost, (int, float)):
        details.append(f"${cost:.6f}")
    if isinstance(seconds, (int, float)):
        details.append(f"{format_duration(float(seconds))} billed audio")
    suffix = f" ({', '.join(details)})" if details else ""
    reporter(f"{prefix}Wrote {output_format} to {destination}{suffix}")


def print_plan(
    label: str,
    source: Path,
    duration: float | None,
    output_path: Path | None,
    output_format: str,
    args: argparse.Namespace,
    reporter: Reporter,
    prefix: str,
) -> None:
    if args.chunk_seconds and duration is not None:
        chunks = max(1, math.ceil(duration / args.chunk_seconds))
    else:
        chunks = 1
    direct = source.suffix.lower() in DIRECT_FORMATS and source.stat().st_size <= DIRECT_SIZE_LIMIT
    conversion = "direct upload" if direct and chunks == 1 else "ffmpeg normalization"
    destination = str(output_path) if output_path is not None else "stdout"
    reporter(
        f"{prefix}{label}: {format_duration(duration)}, {chunks} chunk(s), {conversion}; "
        f"model={args.model}, output={destination} ({output_format})"
    )


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown duration"
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds_value:02d}"
    return f"{minutes}:{seconds_value:02d}"


def format_chunk_span(chunk: Chunk) -> str:
    end = chunk.start + chunk.duration if chunk.duration is not None else None
    return f"{format_duration(chunk.start)}–{format_duration(end)}"


def list_models(args: argparse.Namespace) -> int:
    headers = {"User-Agent": f"openrouter-transcribe-cli/{__version__}"}
    key = os.getenv(args.api_key_env) or os.getenv("OPENROUTER_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(MODELS_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise CliError(f"could not list OpenRouter models: {reason}") from exc
    except json.JSONDecodeError as exc:
        raise CliError("OpenRouter returned invalid model data") from exc

    models = payload.get("data", []) if isinstance(payload, dict) else []
    models = sorted(
        (model for model in models if isinstance(model, dict)),
        key=lambda model: str(model.get("id", "")),
    )
    if args.models_json:
        print(json.dumps(models, ensure_ascii=False, indent=2))
        return 0
    if not models:
        print("No transcription models found.")
        return 0
    width = max(len(str(model.get("id", ""))) for model in models)
    for model in models:
        slug = str(model.get("id", ""))
        name = str(model.get("name", ""))
        marker = " *" if slug == DEFAULT_MODEL else ""
        print(f"{slug:<{width}}  {name}{marker}")
    print("\n* default model")
    return 0
