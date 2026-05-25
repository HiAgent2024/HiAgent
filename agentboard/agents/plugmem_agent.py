"""Context-efficient HiAgent variant with PlugMem recall/upload hooks."""
from __future__ import annotations

import logging
import os
import re
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, List, Optional

from common.registry import registry

from .cme_final import ContextEfficientAgentV2
from .plugmem_client import PlugMemClient


logger = logging.getLogger(__name__)


@registry.register_agent("PlugMemContextEfficientAgent")
class PlugMemContextEfficientAgent(ContextEfficientAgentV2):
    def __init__(
        self,
        llm_model,
        memory_size=100,
        examples=[],
        instruction="",
        init_prompt_path=None,
        system_message="You are a helpful assistant.",
        need_goal=False,
        check_actions=None,
        check_inventory=None,
        use_parser=True,
        plugmem: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            llm_model,
            memory_size,
            examples,
            instruction,
            init_prompt_path,
            system_message,
            need_goal,
            check_actions,
            check_inventory,
            use_parser,
        )
        self.plugmem_config = plugmem or {}
        self.plugmem_enabled = bool(self.plugmem_config.get("enabled", False))
        self.plugmem_recall_on_reset = bool(self.plugmem_config.get("recall_on_reset", True))
        self.plugmem_upload_on_finish = bool(self.plugmem_config.get("upload_on_finish", True))
        self.plugmem_session_id = self.plugmem_config.get("session_id")
        self.plugmem_recall_mode = self.plugmem_config.get("recall_mode", "reason")
        self.plugmem_max_context_chars = int(self.plugmem_config.get("max_context_chars", 700))
        self.plugmem_context = ""
        self.plugmem_client = self._build_plugmem_client()

    def _build_plugmem_client(self) -> Optional[PlugMemClient]:
        if not self.plugmem_enabled:
            return None
        base_url = self.plugmem_config.get("base_url")
        api_key = self.plugmem_config.get("api_key")
        graph_id = self.plugmem_config.get("graph_id")
        if not base_url or not api_key or not graph_id:
            logger.warning("PlugMem is enabled but base_url/api_key/graph_id is incomplete")
            return None
        timeout = self.plugmem_config.get("timeout", 10.0)
        return PlugMemClient(base_url=base_url, api_key=api_key, graph_id=graph_id, timeout=timeout)

    def reset(self, goal, init_obs, init_act=None):
        super().reset(goal, init_obs, init_act)
        self.plugmem_context = ""
        if not self.plugmem_enabled or not self.plugmem_recall_on_reset or self.plugmem_client is None:
            return
        try:
            raw_context = self.plugmem_client.recall(
                observation=init_obs,
                goal=goal,
                task_type=os.environ.get("EVALTASK", ""),
                session_id=self.plugmem_session_id,
                mode=self.plugmem_recall_mode,
            )
            self.plugmem_context = self._filter_plugmem_context(raw_context)
        except Exception as exc:
            self.plugmem_context = ""
            logger.warning("PlugMem recall failed: %s", exc)

    def make_prompt(self, need_goal=False, check_actions="check valid actions", check_inventory="inventory", system_message=''):
        with redirect_stdout(StringIO()):
            prompt = super().make_prompt(
                need_goal=need_goal,
                check_actions=check_actions,
                check_inventory=check_inventory,
                system_message=system_message,
            )
        if self.plugmem_context:
            prompt = self._inject_plugmem_context(prompt)
        print(f'------------[Prompt Start]-----------\n{prompt}\n----------[Prompt END]------------')
        return prompt

    def _inject_plugmem_context(self, prompt: str) -> str:
        block = (
            "Past task hints:\n"
            "Use these as general strategy hints only.\n"
            "Do not copy concrete object names unless they also appear in the current observation.\n\n"
            f"{self.plugmem_context}\n\n"
            "These hints may help with the current task, but the current observation and goal are authoritative.\n\n"
        )
        marker = self._current_history_marker()
        if marker:
            idx = prompt.find(marker)
            if idx >= 0:
                return prompt[:idx] + block + prompt[idx:]
        return block + prompt

    def _filter_plugmem_context(self, text: str) -> str:
        if not text:
            return ""

        context = str(text).strip()
        final_marker = "### Final Information"
        marker_idx = context.find(final_marker)
        if marker_idx >= 0:
            context = context[marker_idx + len(final_marker):]

        lines = []
        for line in context.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append("")
                continue
            if stripped in {"---", "### Reasoning", "### Final Information"}:
                continue
            lines.append(stripped)

        context = "\n".join(lines)
        context = re.sub(r"\n{3,}", "\n\n", context).strip()
        if not context:
            return ""

        max_chars = max(0, self.plugmem_max_context_chars)
        if max_chars and len(context) > max_chars:
            context = context[:max_chars].rstrip()
        return context

    def _current_history_marker(self) -> Optional[str]:
        history = getattr(self, "memory", [])[-self.memory_size:]
        if not history or not history[0]:
            return None
        key, value = history[0][0]
        marker = f"{key}: {value}"
        return marker if value is not None else f"{key}: "

    def remember_current_task(self, task_type: str = "") -> Optional[Dict[str, Any]]:
        if not self.plugmem_enabled or not self.plugmem_upload_on_finish or self.plugmem_client is None:
            return None

        payload = self._memory_to_plugmem_trajectory()
        if payload is None:
            return None

        try:
            return self.plugmem_client.upload_trajectory(
                goal=payload["goal"],
                initial_observation=payload["initial_observation"],
                steps=payload["steps"],
                session_id=self.plugmem_session_id,
            )
        except Exception as exc:
            logger.warning("PlugMem upload failed: %s", exc)
            return None

    def _memory_to_plugmem_trajectory(self) -> Optional[Dict[str, Any]]:
        observations: List[str] = []
        actions: List[str] = []

        for turn in getattr(self, "memory", []):
            for key, value in turn:
                if key == "Observation" and value:
                    observations.append(value)
                elif key == "Action" and value:
                    actions.append(value)

        if not self.goal or not observations or not actions:
            return None

        steps = [
            {"action": actions[i], "observation": observations[i + 1]}
            for i in range(min(len(actions), len(observations) - 1))
        ]
        if not steps:
            return None

        return {
            "goal": self.goal,
            "initial_observation": observations[0],
            "steps": steps,
        }

    @classmethod
    def from_config(cls, llm_model, config):
        memory_size = config.get("memory_size", 100)
        instruction = config.get("instruction", "")
        examples = config.get("examples", [])
        init_prompt_path = config.get("init_prompt_path", None)
        system_message = config.get("system_message", "You are a helpful assistant.")
        check_actions = config.get("check_actions", None)
        check_inventory = config.get("check_inventory", None)
        use_parser = config.get("use_parser", True)
        need_goal = config.get("need_goal", False)
        plugmem = config.get("plugmem", {})
        return cls(
            llm_model,
            memory_size,
            examples,
            instruction,
            init_prompt_path,
            system_message,
            need_goal,
            check_actions,
            check_inventory,
            use_parser,
            plugmem,
        )
