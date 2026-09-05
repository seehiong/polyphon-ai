"""Semantic Structuring and LLM Layer."""

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from polyphon.types import ActionItem, Insights


def extract_json_object(raw_text: str) -> dict | None:
    """Extract and parse a JSON object from raw LLM output, handling markdown, thoughts, and trailing commas."""
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()
    # 1. Remove <think>...</think> blocks if reasoning model emitted thoughts
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Extract from markdown code fences if present: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Try direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Extract outer substring between '{' and '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            relaxed = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(relaxed)
            except json.JSONDecodeError:
                pass

    return None


class LLMBackend(ABC):
    """Abstract interface for local LLM structuring."""

    @abstractmethod
    def extract_insights(self, transcript_md: str) -> Insights:
        """Process markdown dialogue transcript and extract structured meeting intelligence."""

    def infer_speakers(self, snippet: str) -> dict[str, str]:
        """Infer speaker identities and roles from opening dialogue turns."""
        return {}


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment, falling back on bad input."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class OpenAICompatibleLLM(LLMBackend):
    """Local or remote LLM inference via standard OpenAI-compatible API (llama-server, vLLM, Ollama)."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ):
        raw_endpoint = endpoint or os.environ.get("POLYPHON_LLM_ENDPOINT", "http://127.0.0.1:8080/v1")
        if "://localhost:" in raw_endpoint:
            raw_endpoint = raw_endpoint.replace("://localhost:", "://127.0.0.1:")
        self.endpoint = raw_endpoint
        self.model = model or os.environ.get("POLYPHON_LLM_MODEL", "qwen3.8-27b")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        # Summarising a long meeting on CPU or a small GPU can take many minutes,
        # so both limits are configurable rather than fixed.
        self.timeout = timeout if timeout is not None else _env_float("POLYPHON_LLM_TIMEOUT", 900.0)
        self.max_tokens = max_tokens if max_tokens is not None else int(_env_float("POLYPHON_LLM_MAX_TOKENS", 4096))

    def _get_opener(self) -> urllib.request.OpenerDirector:
        """Create an HTTP opener that explicitly bypasses environment proxies for local addresses."""
        is_local = any(h in self.endpoint for h in ("127.0.0.1", "localhost", "0.0.0.0"))
        if is_local:
            return urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return urllib.request.build_opener()

    def infer_speakers(self, snippet: str) -> dict[str, str]:
        """Infer speaker names and roles from opening dialogue turns."""
        system_prompt = (
            "You are an assistant identifying speakers in meeting transcripts.\n"
            "Analyze the opening dialogue turns and identify speaker names and titles if they introduce themselves "
            "or are directly addressed by name.\n"
            "Output ONLY a JSON object mapping speaker IDs to names/roles, e.g.:\n"
            '{"SPEAKER_00": "Sarah (Project Manager)"}\n'
            "If no names are mentioned or cannot be determined, return an empty JSON object: {}."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Opening dialogue:\n\n{snippet}"},
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"},
        }

        url = f"{self.endpoint.rstrip('/')}/chat/completions"
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        opener = self._get_opener()
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            choice = resp_data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            parsed = extract_json_object(content)
            if parsed is None and "reasoning_content" in msg:
                parsed = extract_json_object(msg.get("reasoning_content") or "")
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items() if isinstance(v, str)}
        except Exception:
            pass
        return {}

    def extract_insights(self, transcript_md: str) -> Insights:
        """Prompt local LLM with transcript and extract structured meeting insights as JSON."""
        system_prompt = (
            "You are an expert executive assistant analyzing a multi-speaker meeting transcript.\n"
            "Be concise in any internal thinking. Analyze the conversation and return a valid JSON object matching this schema:\n"
            "{\n"
            '  "summary": "Concise 3-5 sentence executive overview of the meeting.",\n'
            '  "topics": ["Key Topic 1", "Key Topic 2", "Key Topic 3"],\n'
            '  "decisions": ["Agreed decision 1", "Agreed decision 2"],\n'
            '  "action_items": [\n'
            '    {"task": "Actionable task description", "assignee": "Speaker name or null", "due_date": null}\n'
            "  ]\n"
            "}\n"
            "Instructions:\n"
            "- Be concise and focus on concrete decisions and action items.\n"
            "- Do not quote lengthy dialogue inside the JSON.\n"
            "- Return ONLY the valid JSON object without markdown code fences or conversational text."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Here is the dialogue transcript:\n\n{transcript_md}",
                },
            ],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        url = f"{self.endpoint.rstrip('/')}/chat/completions"
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        opener = self._get_opener()
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            choice = resp_data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            parsed = extract_json_object(content)
            if parsed is None and "reasoning_content" in msg:
                parsed = extract_json_object(msg.get("reasoning_content") or "")

            if parsed is None:
                # If the model returned plain text narrative summary instead of JSON, preserve it ONLY if from content
                clean_text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if clean_text:
                    return Insights(
                        summary=clean_text,
                        topics=[],
                        decisions=[],
                        action_items=[],
                    )
                raise ValueError(
                    "Empty or unparseable response from LLM server (token limit may have been reached during reasoning)"
                )

            action_items = []
            for item in parsed.get("action_items", []):
                if isinstance(item, dict) and "task" in item:
                    action_items.append(
                        ActionItem(
                            task=str(item["task"]),
                            assignee=(str(item["assignee"]) if item.get("assignee") else None),
                            due_date=(str(item["due_date"]) if item.get("due_date") else None),
                        )
                    )
                elif isinstance(item, str):
                    action_items.append(ActionItem(task=item))

            return Insights(
                summary=str(parsed.get("summary", "")),
                topics=[str(t) for t in parsed.get("topics", [])],
                decisions=[str(d) for d in parsed.get("decisions", [])],
                action_items=action_items,
            )

        except (urllib.error.URLError, TimeoutError, OSError) as err:
            timed_out = isinstance(err, TimeoutError) or "timed out" in str(err).lower()
            if timed_out:
                detail = (
                    f"⚠️ LLM request to {self.endpoint} timed out after {self.timeout:.0f}s "
                    f"(model '{self.model}'). Long meetings on CPU or a small GPU can exceed this: "
                    "raise it with POLYPHON_LLM_TIMEOUT, or use a smaller/faster model."
                )
            else:
                detail = (
                    f"⚠️ Could not reach the LLM server at {self.endpoint} ({err}). "
                    "Ensure your OpenAI-compatible server is running (llama-server, vLLM, "
                    "Ollama, or LM Studio) and that POLYPHON_LLM_ENDPOINT points at it."
                )
            return Insights(summary=detail, topics=[], decisions=[], action_items=[])
        except (
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as err:
            err_str = str(err)
            if "Empty or unparseable" in err_str:
                detail = (
                    "⚠️ Failed to parse structured insights: LLM returned an empty response. "
                    "If using Ollama, the meeting transcript likely exceeded its default 4096-token "
                    "context window (num_ctx). Raise num_ctx on your model (e.g. PARAMETER num_ctx 16384 in a Modelfile)."
                )
            else:
                detail = f"⚠️ Failed to parse structured insights from LLM response ({err})."
            return Insights(
                summary=detail,
                topics=[],
                decisions=[],
                action_items=[],
            )


class LocalOllamaLLM(LLMBackend):
    """Legacy/Ollama-specific LLM backend."""

    def __init__(self, model: str | None = None, endpoint: str | None = None):
        endpoint = endpoint or os.environ.get("POLYPHON_OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
        model = model or os.environ.get("POLYPHON_LLM_MODEL", "qwen2.5:32b")
        self.model = model
        self.endpoint = endpoint
        self._delegate = OpenAICompatibleLLM(
            endpoint=f"{endpoint.rstrip('/')}/v1",
            model=model,
        )

    def extract_insights(self, transcript_md: str) -> Insights:
        return self._delegate.extract_insights(transcript_md)
