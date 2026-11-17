"""Brain — local-only Ollama inference for the EDXSO sales agent.

Every application model request is sent to the local Ollama server. There is
no hosted LLM provider or cloud fallback in this execution path.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from sellbird.state import StateManager

logger = logging.getLogger("sellbird.brain")

from sellbird.paths import PROJECT_ROOT
from sellbird.textio import read_user_text
from sellbird.workspace_home import current_home

PROMPTS_DIR = PROJECT_ROOT / "prompts"
SKILLS_DIR = PROJECT_ROOT / "skills"

# Local inference timeout and compatibility argument retained for callers.
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MAX_RETRIES = 0


PROVIDER_LABELS = {"ollama": "Ollama"}

# Settings key holding the brain's last health transition, for `sellbird
# status` and the dashboard: {"provider", "status": "ok"|"paused", "reason",
# "since", and for pauses "kind" (plus "until", epoch seconds, for usage limits)}.
BRAIN_STATUS_KEY = "brain_status"


def parse_brain_status(raw: str) -> dict:
    """The stored BRAIN_STATUS_KEY value as a dict ({} if absent or bad)."""
    try:
        value = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def active_brain_status(raw: str, provider: str) -> dict:
    """Return the stored status only when it belongs to the configured provider."""
    status = parse_brain_status(raw)
    return status if status.get("provider") == provider else {}


class BrainUnavailable(Exception):
    """The local inference provider is unavailable."""


def _read_knowledge(folder: str, name: str) -> str:
    """``<home>/<folder>/<name>.md``, or the shipped copy when the home has
    none (a home with no prompts folder, such as a hosted workspace, still
    gets Sellbird's own prompts and skills). Raises FileNotFoundError when
    neither exists."""
    own = getattr(current_home(), f"{folder}_dir") / f"{name}.md"
    try:
        return read_user_text(own)
    except FileNotFoundError:
        from sellbird.home import bundled_defaults
        shipped = bundled_defaults()
        if shipped is None or (shipped / folder / f"{name}.md") == own:
            raise
        return read_user_text(shipped / folder / f"{name}.md")


class Brain:
    def __init__(self, state: StateManager, brain_config=None):
        """Create a Brain whose only inference provider is local Ollama."""
        from sellbird.config import BrainConfig

        self.state = state
        self.config = brain_config or BrainConfig()
        self.provider = self.config.provider
        from sellbird.local_ai.ollama_client import OllamaClient, OllamaSettings
        settings = OllamaSettings.from_env()
        if self.config.model:
            settings = OllamaSettings(
                base_url=settings.base_url,
                model=self.config.model,
                timeout_seconds=settings.timeout_seconds,
            )
        self.ollama = OllamaClient(settings)
        self.model = settings.model
        # Whether the last think_json got no reply at all (an outage, logout,
        # rate limit or timeout), as opposed to a reply that wasn't valid
        # JSON. Callers that count model failures must not count outages.
        self.last_reply_empty = False
        self.unavailable = ""
        self._unavailable_at = 0.0
        self._unavailable_fingerprint = None
        self._usage_limited_until = 0.0
        self._usage_pause_recorded = False
        self._status_synced = False
        # Local inference has no subscription quota endpoint. The app uses
        # its own local call counter for the existing daily-work guard.
        self.quota = None

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS[self.provider]

    @property
    def paused_reason(self) -> str:
        """Why local inference is currently unavailable, or an empty string."""
        return self.unavailable

    async def preflight(self) -> str:
        """Verify the local Ollama server and configured model before inference."""
        from sellbird.local_ai.ollama_client import OllamaError

        try:
            await self.ollama.require_model()
        except OllamaError as error:
            await self._mark_unavailable(error)
            return self.unavailable
        if self.unavailable:
            await self._mark_available()
        elif not self._status_synced:
            logger.info("Ollama ready: local model %s is installed; no cloud inference configured.", self.model)
            self._status_synced = True
            try:
                stored = parse_brain_status(await self.state.get_setting(BRAIN_STATUS_KEY))
            except Exception:
                stored = {}
            if stored.get("status") == "paused":
                await self._mark_available()
        return ""

    def _fingerprint(self):
        return (self.ollama.settings.base_url, self.ollama.settings.model)

    async def _mark_unavailable(self, error):
        first = not self.unavailable or self.unavailable != str(error)
        self.unavailable = str(error)
        self._unavailable_at = time.monotonic()
        self._unavailable_fingerprint = self._fingerprint()
        if first:
            await self._record_status("paused", self.unavailable,
                                      kind=getattr(error, "kind", "setup"))

    async def _mark_available(self):
        logger.info(f"{self.provider_label} is available again; resuming.")
        self.unavailable = ""
        self._unavailable_fingerprint = None
        await self._record_status("ok", "")

    async def _record_status(self, status: str, reason: str, **extra):
        try:
            await self.state.set_setting(BRAIN_STATUS_KEY, json.dumps({
                "provider": self.provider, "status": status, "reason": reason[:500],
                "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **extra,
            }))
        except Exception as e:
            logger.warning(f"Failed to record brain status: {e}")

    async def think(
        self,
        prompt: str,
        session_id: str | None = None,
        expect_json: bool = False,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        agent: str = "",
        task: str = "",
    ) -> str:
        """Generate text exclusively through the configured local Ollama model."""
        if self.provider != "ollama":
            logger.error("Refusing non-local inference provider: %s", self.provider)
            return ""
        response = await self._think_ollama(
            prompt, expect_json=expect_json, timeout=timeout, agent=agent, task=task
        )
        if not response and self.unavailable:
            # Keep the exact request in SQLite so it can be retried after the
            # local Ollama service/model becomes available again. No cloud
            # provider is consulted and callers retain the existing empty
            # response behaviour while the task waits.
            enqueue = getattr(self.state, "enqueue_local_ai_task", None)
            if enqueue is not None:
                try:
                    queued_id = await enqueue(
                        prompt, agent=agent, task=task, expect_json=expect_json
                    )
                    logger.warning("LOCAL AI task queued id=%s agent=%s task=%s",
                                   queued_id, agent, task)
                except Exception as error:
                    logger.error("Could not queue failed local AI task: %s", error)
        return response

    async def think_json(
        self,
        prompt: str,
        session_id: str | None = None,
        agent: str = "",
        task: str = "",
    ) -> dict | list | None:
        """Send a prompt and parse the response as JSON."""
        full_prompt = (
            prompt
            + "\n\nRespond ONLY with valid JSON. No markdown, no explanation."
        )
        response = await self.think(
            full_prompt, session_id=session_id, expect_json=True, agent=agent, task=task
        )
        self.last_reply_empty = not response
        if not response:
            return None
        parsed = self._extract_json(response)
        if parsed is None:
            logger.error(f"Failed to parse JSON from brain: {response[:200]}")
        return parsed

    @staticmethod
    def _extract_json(text: str) -> dict | list | None:
        """Best-effort JSON extraction: strips code fences, then falls back to
        locating the outermost JSON object/array in surrounding prose."""
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Fall back: model wrapped the JSON in explanation text
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = cleaned.find(open_ch)
            end = cleaned.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    continue
        return None

    async def check_usage(self) -> float:
        """Check today's usage on this Brain's provider (own call count).

        Returns the number of calls made today. Errors are treated as
        "over budget is unknown" and return a safe 0.0 so a broken DB read
        doesn't crash the heartbeat.
        """
        try:
            return await self.state.get_usage_today(provider=self.provider)
        except Exception as e:
            logger.warning(f"Could not read usage from state: {e}")
            return 0.0

    async def is_within_budget(
        self, max_daily_calls: int = 200, max_percent: float | None = None
    ) -> bool:
        """Apply the local daily inference-call limit; no subscription quota is queried."""
        if max_percent is not None and self.quota is not None:
            try:
                windows = await self.quota.get_utilization()
            except Exception as e:
                logger.debug(f"Quota check errored: {e}")
                windows = None
            if windows:
                worst = max(
                    (w["utilization"] for w in windows.values()), default=0.0
                )
                within = worst < max_percent
                if not within:
                    resets = ", ".join(
                        f"{name} resets {w['resets_at']}"
                        for name, w in windows.items()
                        if w.get("resets_at")
                    )
                    logger.info(
                        f"{self.provider_label} quota gate: utilization {worst:.0f}% >= "
                        f"{max_percent:.0f}% limit. {resets}"
                    )
                return within

        calls = await self.check_usage()
        return calls < max_daily_calls

    async def _think_ollama(
        self, prompt: str, *, expect_json: bool, timeout: float, agent: str, task: str
    ) -> str:
        """Run one inference request against the configured local Ollama model."""
        from sellbird.local_ai.ollama_client import OllamaError

        started = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self.ollama.generate(prompt, expect_json=expect_json), timeout=timeout
            )
        except asyncio.CancelledError:
            raise
        except (OllamaError, asyncio.TimeoutError) as error:
            logger.error("Local Ollama inference unavailable (%s/%s): %s", agent, task, error)
            await self._mark_unavailable(error)
            return ""
        except Exception as error:
            logger.error("Local Ollama inference failed (%s/%s): %s", agent, task, type(error).__name__)
            await self._mark_unavailable(error)
            return ""

        if self.unavailable:
            await self._mark_available()
        duration_ms = int((time.monotonic() - started) * 1000)
        try:
            await self.state.increment_usage(provider="ollama")
        except Exception as error:
            logger.warning("Failed to record local inference counter: %s", error)
        try:
            await self.state.record_usage_event(
                agent=agent,
                task=task,
                model=self.model,
                duration_ms=duration_ms,
                is_error=False,
                source="ollama_local",
                provider="ollama",
            )
        except Exception as error:
            logger.warning("Failed to record local inference event: %s", error)
        logger.info(
            "LOCAL INFERENCE completed provider=ollama model=%s agent=%s task=%s duration_ms=%s",
            self.model, agent, task, duration_ms,
        )
        return response

    async def process_queued_tasks(self, limit: int = 10) -> dict:
        """Retry pending inference requests after local Ollama is restored.

        Results are stored in SQLite for inspection. This method never sends
        queued prompts to a hosted service. A failed retry returns the task to
        pending state so a later invocation can try again.
        """
        pending_reader = getattr(self.state, "get_pending_local_ai_tasks", None)
        updater = getattr(self.state, "update_local_ai_task", None)
        if pending_reader is None or updater is None:
            return {"processed": 0, "completed": 0, "pending": 0,
                    "error": "Persistent local AI queue is unavailable."}
        rows = await pending_reader(limit=limit)
        if not rows:
            return {"processed": 0, "completed": 0, "pending": 0, "error": ""}
        problem = await self.preflight()
        if problem:
            return {"processed": 0, "completed": 0, "pending": len(rows),
                    "error": problem}
        completed = 0
        processed = 0
        for row in rows:
            task_id = row["id"]
            await updater(task_id, status="processing")
            processed += 1
            result = await self._think_ollama(
                row["prompt"], expect_json=bool(row.get("expect_json")),
                timeout=DEFAULT_TIMEOUT_SECONDS, agent=row.get("agent", "queued"),
                task=row.get("task", "retry"),
            )
            if result:
                await updater(task_id, status="completed", result_text=result)
                completed += 1
            else:
                await updater(task_id, status="pending", error=self.paused_reason or
                              "Local inference returned no response.")
                break
        remaining = await pending_reader(limit=100)
        logger.info("LOCAL AI queue retry processed=%s completed=%s pending=%s",
                    processed, completed, len(remaining))
        return {"processed": processed, "completed": completed,
                "pending": len(remaining), "error": self.paused_reason}

    def load_prompt(self, prompt_name: str, **kwargs) -> str:
        """Load a prompt template from the prompts/ directory and fill in variables."""
        try:
            template = _read_knowledge("prompts", prompt_name)
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {prompt_name}.md")
            return ""
        except OSError as e:
            logger.error(f"Could not read prompt file {prompt_name}.md: {e}")
            return ""
        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
        # Surface templating mistakes early instead of sending {{foo}} to the model
        leftover = re.findall(r"\{\{(\w+)\}\}", template)
        if leftover:
            logger.warning(
                f"Prompt '{prompt_name}' has unfilled variables: {sorted(set(leftover))}"
            )
        return template

    def load_skill(self, skill_name: str) -> str:
        """Load a skill knowledge file from the skills/ directory."""
        try:
            return _read_knowledge("skills", skill_name)
        except FileNotFoundError:
            logger.warning(f"Skill file not found: {skill_name}.md")
            return ""
        except OSError as e:
            logger.error(f"Could not read skill file {skill_name}.md: {e}")
            return ""

    def load_skills_for_agent(self, agent_name: str) -> str:
        """Load all relevant skills for a specific sub-agent.

        Returns concatenated skill content that should be injected
        into the agent's prompts as foundational knowledge.
        """
        # product_knowledge and competitive_intel are auto-generated by the trainer
        # and loaded by every agent that needs product context
        skill_map = {
            "scout": ["prospecting_tactics", "lead_qualification", "account_navigation", "signal_playbook", "product_knowledge"],
            # The writer gets a short prompt (prompts/writer.md) and only the part of
            # signal_playbook that fits each prospect (Writer._signal_guide): long
            # playbooks bury the few rules that matter for a 40-word email.
            "writer": ["product_knowledge", "competitive_intel"],
            "handler": ["objection_handling", "sales_methodology", "offer_strategy", "product_knowledge", "competitive_intel"],
            "sender": ["email_frameworks", "product_knowledge"],
            "linkedin": ["linkedin_outreach", "prospecting_tactics", "product_knowledge"],
        }

        skill_names = skill_map.get(agent_name, [])
        if not skill_names:
            return ""

        sections = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                sections.append(content)

        if not sections:
            return ""

        return (
            "\n\n---\n## FOUNDATIONAL KNOWLEDGE\n"
            "Use the following frameworks and best practices to guide your work:\n\n"
            + "\n\n---\n\n".join(sections)
        )
