from __future__ import annotations

import json
import urllib.error
import urllib.request

from phase2_agent.candidate_space import CandidateConfig
from phase2_agent.config import LLMSettings


class LLMSuggester:
    def __init__(self, settings: LLMSettings):
        self.settings = settings

    def suggest(self, history: list[dict]) -> list[CandidateConfig]:
        if not self.settings.enabled:
            return []
        prompt = {
            "task": "Suggest up to 2 next LoRA kernel search candidates.",
            "constraints": {
                "strategies": ["cublas"],
                "main_backend": ["sgemm", "gemm_ex_tf32"],
                "low_rank_backend": ["sgemm", "gemm_ex_tf32"],
                "accumulation_order": ["wx_then_lora", "lora_then_wx"],
                "allow_tf32": [True, False],
            },
            "history": history[-5:],
            "return_json_schema": {
                "candidates": [
                    {
                        "strategy": "cublas",
                        "main_backend": "gemm_ex_tf32",
                        "low_rank_backend": "sgemm",
                        "accumulation_order": "wx_then_lora",
                        "allow_tf32": True,
                        "notes": "why this candidate is worth trying"
                    }
                ]
            },
        }
        body = {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "You are helping a CUDA search agent choose the next parameterized candidate. "
                        "Return strict JSON only.\n" + json.dumps(prompt, ensure_ascii=True)
                    ),
                }
            ],
            "temperature": 0.2,
            "max_completion_tokens": 400,
        }
        req = urllib.request.Request(
            self.settings.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return []

        try:
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return []

        candidates = []
        for item in parsed.get("candidates", [])[:2]:
            try:
                candidates.append(
                    CandidateConfig(
                        strategy=item["strategy"],
                        main_backend=item["main_backend"],
                        low_rank_backend=item["low_rank_backend"],
                        accumulation_order=item["accumulation_order"],
                        allow_tf32=bool(item["allow_tf32"]),
                        notes=item.get("notes", "LLM-suggested candidate"),
                    )
                )
            except KeyError:
                continue
        return candidates

