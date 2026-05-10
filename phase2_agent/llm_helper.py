from __future__ import annotations

import json
import urllib.error
import urllib.request

from phase2_agent.candidate_space import CandidateConfig
from phase2_agent.config import LLMSettings
from phase2_agent.tracing import TraceLogger


class LLMSuggester:
    def __init__(self, settings: LLMSettings, tracer: TraceLogger):
        self.settings = settings
        self.tracer = tracer

    def _candidate_urls(self) -> list[str]:
        base = self.settings.base_url.rstrip("/")
        urls = [base + "/chat/completions"]
        if not base.endswith("/v1"):
            urls.append(base + "/v1/chat/completions")
        return urls

    def suggest(self, history: list[dict]) -> list[CandidateConfig]:
        if not self.settings.enabled:
            self.tracer.log("llm_skipped", reason="llm_disabled")
            return []
        prompt = {
            "task": "Suggest up to 2 next LoRA candidates for a 2D float32 CUDA workload where recent evidence favors ATen/cuBLAS compositions over custom caching tricks.",
            "constraints": {
                "strategies": ["aten"],
                "main_backend": [
                    "addmm",
                    "addmm_inplace",
                    "mmout_addmm_inplace",
                    "mmout_addmm_out",
                    "separate_add_inplace",
                    "static_overlap_out",
                ],
                "low_rank_backend": ["bt_contiguous", "bt_contiguous_out", "bt_strided"],
                "accumulation_order": ["mainfirst", "lorafirst"],
                "allow_tf32": [False],
                "cache_mode": ["none"],
            },
            "history": history[-5:],
            "return_json_schema": {
                "candidates": [
                    {
                        "strategy": "aten",
                        "main_backend": "mmout_addmm_inplace",
                        "low_rank_backend": "bt_contiguous",
                        "accumulation_order": "mainfirst",
                        "allow_tf32": False,
                        "cache_mode": "none",
                        "variant_name": "aten_candidate_name",
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
        payload = None
        last_error = ""
        for url in self._candidate_urls():
            req = urllib.request.Request(
                url,
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
                    self.tracer.log("llm_response_received", url=url, keys=list(payload.keys()))
                    break
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                self.tracer.log("llm_request_attempt_failed", url=url, error=last_error)
        if payload is None:
            self.tracer.log("llm_request_failed", error=last_error or "unknown_error")
            return []

        try:
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            self.tracer.log("llm_response_parsed", raw_length=len(content))
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            self.tracer.log("llm_parse_failed", error=str(exc))
            return []

        candidates = []
        for item in parsed.get("candidates", [])[:2]:
            try:
                candidates.append(
                    CandidateConfig(
                        strategy=item.get("strategy", "aten"),
                        main_backend=item["main_backend"],
                        low_rank_backend=item.get("low_rank_backend", "bt_contiguous"),
                        accumulation_order=item.get("accumulation_order", "mainfirst"),
                        allow_tf32=bool(item.get("allow_tf32", False)),
                        cache_mode=item.get("cache_mode", "none"),
                        variant_name=item.get("variant_name", ""),
                        notes=item.get("notes", "LLM-suggested candidate"),
                    )
                )
            except KeyError:
                continue
        self.tracer.log("llm_candidates_proposed", count=len(candidates))
        return candidates
