"""Lightweight HTTP client for the PlugMem API."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib import error, request


class PlugMemClient:
    def __init__(self, base_url: str, api_key: str, graph_id: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.graph_id = graph_id
        self.timeout = timeout

    def recall(
        self,
        observation: str,
        goal: Optional[str],
        task_type: str = "",
        session_id: Optional[str] = None,
        mode: str = "reason",
    ) -> str:
        payload: Dict[str, Any] = {
            "observation": observation,
            "goal": goal,
            "task_type": task_type,
        }
        if session_id:
            payload["session_id"] = session_id
        if mode == "recall_text":
            return self._post_text(f"/api/v1/graphs/{self.graph_id}/recall_text", payload)
        data = self._post(f"/api/v1/graphs/{self.graph_id}/reason", payload)
        return data.get("reasoning", "")

    def upload_trajectory(
        self,
        goal: str,
        initial_observation: str,
        steps: List[Dict[str, str]],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": "trajectory",
            "goal": goal,
            "initial_observation": initial_observation,
            "steps": steps,
        }
        if session_id:
            payload["session_id"] = session_id
        return self._post(f"/api/v1/graphs/{self.graph_id}/memories", payload)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response_body = self._post_raw(path, payload)
        if not response_body:
            return {}
        return json.loads(response_body)

    def _post_text(self, path: str, payload: Dict[str, Any]) -> str:
        return self._post_raw(path, payload).strip()

    def _post_raw(self, path: str, payload: Dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url + path,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"PlugMem request failed with HTTP {exc.code}: {response_body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"PlugMem request failed: {exc}") from exc
