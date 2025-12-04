"""Tests for sweep eval integration."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from foundry.cli.sweep import run_eval_on_checkpoint


def test_run_eval_on_checkpoint():
    """Run eval on checkpoint returns score."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "ckpt.pt"
        checkpoint_path.write_text("dummy")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="0.75\n")

            score = run_eval_on_checkpoint(checkpoint_path, "gsm8k")

            assert score == 0.75


def test_run_eval_on_checkpoint_failure():
    """Handle eval failure gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "ckpt.pt"
        checkpoint_path.write_text("dummy")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")

            score = run_eval_on_checkpoint(checkpoint_path, "gsm8k")

            assert score == 0.0


def test_run_eval_on_checkpoint_timeout():
    """Handle eval timeout gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "ckpt.pt"
        checkpoint_path.write_text("dummy")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("timeout")

            score = run_eval_on_checkpoint(checkpoint_path, "gsm8k")

            assert score == 0.0
