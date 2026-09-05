"""Unit tests for Polyphon Meeting Bot."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from polyphon.bot.meeting_bot import AUDIO_INTERCEPTOR_SCRIPT, MeetingBot


def test_meeting_bot_initialization():
    bot = MeetingBot(
        meeting_url="https://meet.jit.si/polyphon-test-room",
        bot_name="Custom Notetaker",
        server_ws_url="ws://127.0.0.1:7860/api/stream/ws",
        headless=True,
        duration_mins=15,
    )
    assert bot.meeting_url == "https://meet.jit.si/polyphon-test-room"
    assert bot.bot_name == "Custom Notetaker"
    assert bot.server_ws_url == "ws://127.0.0.1:7860/api/stream/ws"
    assert bot.headless is True
    assert bot.duration_mins == 15
    assert bot.is_running is False
    assert "window.__polyphon_audio_hook_installed" in AUDIO_INTERCEPTOR_SCRIPT


@pytest.mark.anyio
async def test_meeting_bot_stop_lifecycle():
    statuses = []
    bot = MeetingBot(
        meeting_url="https://meet.google.com/abc-defg-hij",
        on_status=lambda s: statuses.append(s),
    )
    mock_browser = AsyncMock()
    mock_page = AsyncMock()
    mock_playwright = AsyncMock()
    bot._browser = mock_browser
    bot._page = mock_page
    bot._playwright = mock_playwright
    bot.is_running = True

    await bot.stop()

    assert bot.is_running is False
    assert bot._browser is None
    assert bot._page is None
    assert bot._playwright is None
    mock_page.evaluate.assert_awaited_once()
    mock_browser.close.assert_awaited_once()
    mock_playwright.stop.assert_awaited_once()


@pytest.mark.anyio
async def test_meeting_bot_jitsi_navigation_mocked():
    bot = MeetingBot(
        meeting_url="https://meet.jit.si/test-room",
        bot_name="Polyphon Bot",
        headless=True,
    )

    mock_input = MagicMock()
    mock_input.count = AsyncMock(return_value=1)
    mock_first_input = MagicMock()
    mock_first_input.is_visible = AsyncMock(return_value=True)
    mock_first_input.fill = AsyncMock()
    mock_input.first = mock_first_input

    mock_btn = MagicMock()
    mock_btn.count = AsyncMock(return_value=1)
    mock_first_btn = MagicMock()
    mock_first_btn.is_visible = AsyncMock(return_value=True)
    mock_first_btn.click = AsyncMock()
    mock_btn.first = mock_first_btn

    mock_page = MagicMock()
    mock_page.locator = MagicMock(side_effect=lambda sel: mock_input if "input" in sel else mock_btn)

    bot._page = mock_page

    await bot._handle_jitsi_join()

    mock_first_input.fill.assert_awaited_with("Polyphon Bot")
    mock_first_btn.click.assert_awaited_once()
