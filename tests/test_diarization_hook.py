from unittest.mock import MagicMock

from polyphon.diarization.base import PolyphonDiarizationHook


def test_diarization_hook_lifecycle_and_throttling():
    hook = PolyphonDiarizationHook(min_refresh_interval=0.5)

    with hook:
        assert hook.progress is not None
        mock_progress = MagicMock()
        hook.progress = mock_progress

        # Step 1: initial update
        hook("segmentation", total=100, completed=10)
        assert mock_progress.update.call_count == 1

        # Step 2: immediate subsequent call within 0.5s should be throttled
        hook("segmentation", total=100, completed=11)
        assert mock_progress.update.call_count == 1

        # Step 3: completion should bypass throttling
        hook("segmentation", total=100, completed=100)
        assert mock_progress.update.call_count == 2

        # Step 4: change of step name should immediately trigger update
        hook("embeddings", total=500, completed=1)
        assert mock_progress.update.call_count == 3
