from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from transcribe_cli.cli import (
    Chunk,
    ChunkResult,
    combine_results,
    format_timestamp,
    infer_output_format,
    main,
    render_transcript,
)


class RenderingTests(unittest.TestCase):
    def test_format_inference(self) -> None:
        self.assertEqual(infer_output_format(Path("captions.srt"), None), "srt")
        self.assertEqual(infer_output_format(Path("captions.unknown"), None), "txt")
        self.assertEqual(infer_output_format(None, "json"), "json")

    def test_timestamp_rounding(self) -> None:
        self.assertEqual(format_timestamp(3661.2346, ","), "01:01:01,235")

    def test_chunk_timestamps_are_offset(self) -> None:
        results = [
            ChunkResult(
                Chunk(Path("one.ogg"), 1, 2, 0.0, 10.0),
                {"text": "Hello", "segments": [{"start": 1.0, "end": 2.0, "text": "Hello"}]},
                "gen-one",
            ),
            ChunkResult(
                Chunk(Path("two.ogg"), 2, 2, 10.0, 8.0),
                {
                    "text": "world",
                    "language": "en",
                    "segments": [{"start": 0.5, "end": 1.5, "text": "world"}],
                    "usage": {"cost": 0.01, "seconds": 8},
                },
                "gen-two",
            ),
        ]
        transcript = combine_results(
            results,
            source_label="sample.wav",
            source_duration=18.0,
            model="example/model",
            requested_language=None,
        )
        self.assertEqual(transcript.text, "Hello\nworld")
        self.assertEqual(transcript.segments[1]["start"], 10.5)
        self.assertEqual(transcript.language, "en")
        self.assertEqual(transcript.usage["cost"], 0.01)
        self.assertIn("00:00:10,500 --> 00:00:11,500", render_transcript(transcript, "srt"))


class _Handler(BaseHTTPRequestHandler):
    body = b""
    authorization = ""

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        type(self).body = self.rfile.read(length)
        type(self).authorization = self.headers.get("Authorization", "")
        payload = json.dumps({"text": "Hosted transcription works.", "usage": {"cost": 0.0001}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Generation-Id", "generation-test")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


class EndToEndTests(unittest.TestCase):
    def test_cli_uploads_file_and_writes_text(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                audio = root / "voice.wav"
                output = root / "transcript.txt"
                audio.write_bytes(b"RIFF fake wave data")
                url = f"http://127.0.0.1:{server.server_port}/transcriptions"
                with (
                    mock.patch.dict(os.environ, {"OPENROUTER_TRANSCRIPTION_KEY": "test-key"}),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = main([str(audio), "--api-url", url, "-o", str(output), "-q"])
                self.assertEqual(result, 0)
                self.assertEqual(output.read_text(), "Hosted transcription works.\n")
                self.assertEqual(_Handler.authorization, "Bearer test-key")
                self.assertIn(b'name="model"', _Handler.body)
                self.assertIn(b'name="file"; filename="voice.wav"', _Handler.body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
