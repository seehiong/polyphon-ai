import json
import urllib.error
from unittest.mock import MagicMock, patch

from polyphon.llm.base import OpenAICompatibleLLM
from polyphon.types import ActionItem, Insights


def test_llm_defaults(monkeypatch):
    # A developer's local .env is loaded at import time and would otherwise
    # override these defaults, so clear the variables for this assertion.
    monkeypatch.delenv("POLYPHON_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("POLYPHON_LLM_MODEL", raising=False)
    llm = OpenAICompatibleLLM()
    assert llm.endpoint == "http://127.0.0.1:8080/v1"
    assert llm.model == "qwen3.8-27b"


def test_llm_extract_insights_success():
    llm = OpenAICompatibleLLM(endpoint="http://127.0.0.1:8080/v1", model="test-model")

    mock_llm_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "summary": "Discussed AI open-source releases.",
                            "topics": ["Llama", "Open Source AI"],
                            "decisions": ["Release model with open weights"],
                            "action_items": [
                                {
                                    "task": "Prepare release notes",
                                    "assignee": "Mark",
                                    "due_date": "Friday",
                                }
                            ],
                        }
                    )
                }
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(mock_llm_payload).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.OpenerDirector.open", return_value=mock_response):
        insights = llm.extract_insights("# Transcript\n[00:00] Alice: Hi.")

    assert isinstance(insights, Insights)
    assert insights.summary == "Discussed AI open-source releases."
    assert "Llama" in insights.topics
    assert "Release model with open weights" in insights.decisions
    assert len(insights.action_items) == 1
    assert insights.action_items[0].task == "Prepare release notes"
    assert insights.action_items[0].assignee == "Mark"
    assert insights.action_items[0].due_date == "Friday"


def test_llm_extract_insights_markdown_fenced():
    llm = OpenAICompatibleLLM()

    raw_json = json.dumps(
        {
            "summary": "Fenced summary test.",
            "topics": ["Fencing"],
            "decisions": [],
            "action_items": ["Action 1"],
        }
    )

    fenced_content = f"```json\n{raw_json}\n```"

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"choices": [{"message": {"content": fenced_content}}]}).encode(
        "utf-8"
    )
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.OpenerDirector.open", return_value=mock_response):
        insights = llm.extract_insights("dummy transcript")

    assert insights.summary == "Fenced summary test."
    assert insights.topics == ["Fencing"]
    assert len(insights.action_items) == 1
    assert isinstance(insights.action_items[0], ActionItem)
    assert insights.action_items[0].task == "Action 1"


def test_llm_connection_failure_fallback():
    llm = OpenAICompatibleLLM(endpoint="http://localhost:9999/v1")

    with patch(
        "urllib.request.OpenerDirector.open",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        insights = llm.extract_insights("dummy transcript")

    assert isinstance(insights, Insights)
    assert "Could not reach the LLM server" in insights.summary
    assert insights.topics == []
    assert insights.action_items == []


def test_llm_reasoning_and_trailing_commas():
    llm = OpenAICompatibleLLM()

    raw_response = (
        "<think>Let me evaluate the design discussion.\nThe team talked about remote controls.</think>\n"
        "Here is the parsed JSON output:\n"
        '{\n  "summary": "Team designed remote control with flip-top LCD.",\n  "topics": ["Hardware", "Pricing",],\n'
        '  "decisions": ["Keep selling price at 25 euros"],\n  "action_items": [{"task": "Order plastic parts"}]\n}'
    )

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"choices": [{"message": {"content": raw_response}}]}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.OpenerDirector.open", return_value=mock_response):
        insights = llm.extract_insights("dummy transcript")

    assert insights.summary == "Team designed remote control with flip-top LCD."
    assert "Hardware" in insights.topics
    assert "Keep selling price at 25 euros" in insights.decisions
    assert insights.action_items[0].task == "Order plastic parts"


def test_llm_narrative_text_fallback():
    llm = OpenAICompatibleLLM()

    raw_response = "The participants discussed remote control ergonomics and agreed on a 25 euro price."

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"choices": [{"message": {"content": raw_response}}]}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.OpenerDirector.open", return_value=mock_response):
        insights = llm.extract_insights("dummy transcript")

    assert insights.summary == "The participants discussed remote control ergonomics and agreed on a 25 euro price."
    assert insights.topics == []


def test_llm_timeout_configurable_from_env(monkeypatch):
    monkeypatch.delenv("POLYPHON_LLM_TIMEOUT", raising=False)
    assert OpenAICompatibleLLM().timeout == 900.0

    monkeypatch.setenv("POLYPHON_LLM_TIMEOUT", "1800")
    assert OpenAICompatibleLLM().timeout == 1800.0

    # Unusable values must not disable the timeout entirely.
    for bad in ("garbage", "0", "-5"):
        monkeypatch.setenv("POLYPHON_LLM_TIMEOUT", bad)
        assert OpenAICompatibleLLM().timeout == 900.0


def test_llm_timeout_message_distinguishes_from_unreachable():
    llm = OpenAICompatibleLLM(endpoint="http://127.0.0.1:8080/v1", model="test-model")

    with patch("urllib.request.OpenerDirector.open", side_effect=TimeoutError("timed out")):
        summary = llm.extract_insights("dummy").summary
    assert "timed out" in summary
    assert "POLYPHON_LLM_TIMEOUT" in summary

    with patch(
        "urllib.request.OpenerDirector.open",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        summary = llm.extract_insights("dummy").summary
    assert "Could not reach" in summary
    # Must not name one specific server; any OpenAI-compatible one works.
    assert "llama-server-rocm" not in summary
    assert "Ollama" in summary


def test_llm_empty_response_explains_ollama_context_overflow():
    llm = OpenAICompatibleLLM()
    mock_response = MagicMock()
    # When Ollama context is exceeded, message content is empty
    mock_response.read.return_value = json.dumps({"choices": [{"message": {"content": ""}}]}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.OpenerDirector.open", return_value=mock_response):
        summary = llm.extract_insights("dummy transcript").summary

    assert "empty response" in summary
    assert "Ollama" in summary
    assert "num_ctx" in summary
