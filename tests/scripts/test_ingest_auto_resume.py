"""Tests for scripts/ingest.py --auto-resume.

Two layers:
  - Mocked subprocess tests exercise the resume loop's control flow (attempt
    counting, stall detection, success/give-up conditions) deterministically,
    without spawning real processes or waiting on a real crash.
  - One real end-to-end test spawns the actual CLI against a small real data
    folder to prove the subprocess wiring itself works: argv construction,
    checkpoint-dir plumbing, and exit-code handling all connect correctly.

What this suite does NOT prove: that restarting genuinely survives a real
OS-level OOM kill. That mechanism (spawn a fresh process, check its exit
code, restart if non-zero) was exercised for real earlier by an equivalent
bash wrapper driving the full 64-file corpus through several real memory
crashes to completion; this Python implementation automates the same
generic, cause-agnostic mechanism.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import ingest

ROOT = Path(__file__).resolve().parents[2]
TRUE_DATA = ROOT / "DATA" / "true_data"


def _args(tmp_path: Path, **overrides) -> ingest.argparse.Namespace:
    defaults = dict(
        source_dir=tmp_path / "source",
        collection="rag_documents",
        chunk_size=1200,
        chunk_overlap=150,
        dry_run=True,
        checkpoint_dir=tmp_path / "checkpoint",
        no_checkpoint=False,
        auto_resume=True,
        max_resume_attempts=5,
    )
    defaults.update(overrides)
    return ingest.argparse.Namespace(**defaults)


class AutoResumeControlFlowTests(unittest.TestCase):
    """Resume-loop logic with subprocess.run mocked out."""

    def test_returns_zero_on_first_successful_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            with patch.object(
                ingest.subprocess, "run", return_value=MagicMock(returncode=0)
            ) as run:
                exit_code = ingest.run_auto_resume(args)

            self.assertEqual(0, exit_code)
            run.assert_called_once()

    def test_retries_after_crash_and_then_succeeds(self) -> None:
        # A crash that still banks new checkpoint files must be retried, not
        # treated as a stall -- only *zero* new progress should stop the loop.
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            args = _args(Path(tmp), checkpoint_dir=checkpoint_dir)

            call_count = 0

            def fake_run(argv, **kwargs):
                nonlocal call_count
                call_count += 1
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                (checkpoint_dir / f"file_{call_count}.json").write_text("{}")
                # Crashes (killed process) on the first attempt, then succeeds.
                return MagicMock(returncode=0 if call_count >= 2 else 1)

            with patch.object(ingest.subprocess, "run", side_effect=fake_run):
                exit_code = ingest.run_auto_resume(args)

            self.assertEqual(0, exit_code)
            self.assertEqual(2, call_count)

    def test_stalls_stop_the_loop_instead_of_retrying_forever(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            args = _args(Path(tmp), checkpoint_dir=checkpoint_dir, max_resume_attempts=10)

            call_count = 0

            def fake_run(argv, **kwargs):
                nonlocal call_count
                call_count += 1
                # Same file kills every attempt; no checkpoint growth, ever.
                return MagicMock(returncode=1)

            with patch.object(ingest.subprocess, "run", side_effect=fake_run):
                exit_code = ingest.run_auto_resume(args)

            self.assertEqual(1, exit_code)
            self.assertEqual(1, call_count)

    def test_gives_up_after_max_attempts_when_still_progressing(self) -> None:
        # Genuine forward progress every attempt, but never actually finishes
        # (returncode never 0) -- must not loop past the configured cap.
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            args = _args(Path(tmp), checkpoint_dir=checkpoint_dir, max_resume_attempts=3)

            call_count = 0

            def fake_run(argv, **kwargs):
                nonlocal call_count
                call_count += 1
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                (checkpoint_dir / f"file_{call_count}.json").write_text("{}")
                return MagicMock(returncode=1)

            with patch.object(ingest.subprocess, "run", side_effect=fake_run):
                exit_code = ingest.run_auto_resume(args)

            self.assertEqual(1, exit_code)
            self.assertEqual(3, call_count)

    def test_child_argv_uses_same_python_and_forwards_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            args = _args(
                Path(tmp),
                checkpoint_dir=checkpoint_dir,
                collection="custom_collection",
                chunk_size=800,
                chunk_overlap=50,
            )

            with patch.object(
                ingest.subprocess, "run", return_value=MagicMock(returncode=0)
            ) as run:
                ingest.run_auto_resume(args)

            argv = run.call_args[0][0]
            self.assertEqual(sys.executable, argv[0])
            self.assertIn("--collection", argv)
            self.assertEqual("custom_collection", argv[argv.index("--collection") + 1])
            self.assertEqual("800", argv[argv.index("--chunk-size") + 1])
            self.assertEqual("50", argv[argv.index("--chunk-overlap") + 1])
            self.assertIn("--dry-run", argv)
            # The child must never itself receive --auto-resume, or every
            # attempt would recursively spawn its own resume loop.
            self.assertNotIn("--auto-resume", argv)

    def test_auto_resume_with_no_checkpoint_is_rejected_by_argparse(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "ingest.py"),
             str(TRUE_DATA), "--auto-resume", "--no-checkpoint"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("--auto-resume requires checkpointing", result.stderr)


class AutoResumeRealSubprocessTest(unittest.TestCase):
    """One real end-to-end run of the actual CLI, no mocking."""

    def test_real_run_completes_and_populates_checkpoint(self) -> None:
        source_files = sorted(TRUE_DATA.glob("*"))[:3]
        self.assertGreaterEqual(len(source_files), 1, "expected sample files in DATA/true_data")

        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            source_dir.mkdir()
            for path in source_files:
                if path.is_file():
                    shutil.copy2(path, source_dir / path.name)
            checkpoint_dir = Path(tmp) / "checkpoint"

            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "ingest.py"),
                    str(source_dir), "--dry-run", "--auto-resume",
                    "--checkpoint-dir", str(checkpoint_dir),
                ],
                capture_output=True, text=True, timeout=120,
            )

            self.assertEqual(0, result.returncode, msg=result.stdout + result.stderr)
            self.assertIn("auto-resume: completed on attempt 1", result.stdout)
            cached = list(checkpoint_dir.glob("*.json"))
            self.assertGreaterEqual(len(cached), 1)

            # Second real invocation: unrelated to crash-recovery, but proves
            # the checkpoint dir this wrapper wires through is the same one
            # ingest_folder reads from, i.e. a genuine resume would work.
            second = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "ingest.py"),
                    str(source_dir), "--dry-run", "--auto-resume",
                    "--checkpoint-dir", str(checkpoint_dir),
                ],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(0, second.returncode, msg=second.stdout + second.stderr)
            self.assertIn(f"({len(cached)} from checkpoint)", second.stdout)


if __name__ == "__main__":
    unittest.main()
