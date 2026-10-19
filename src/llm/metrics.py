import time
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class CallMetric:
    call_id: int
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    retries: int
    finish_reason: str
    success: bool
    motivo: str = "ok"

class LLMMetricsManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls: List[CallMetric] = []; self._counter = 0

    def registrar(self, prompt_tokens: int, completion_tokens: int, latency: float, retries: int, finish_reason: str, success: bool, motivo: str = "ok"):
        self._counter += 1
        metric = CallMetric(
            call_id=self._counter,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency,
            retries=retries,
            finish_reason=finish_reason,
            success=success,
            motivo=motivo
        )
        self.calls.append(metric)

    def summary(self) -> Dict[str, Any]:
        total_calls = len(self.calls)
        if total_calls == 0: return {"total_calls": 0, "avg_latency": 0.0, "throughput": 0.0, "total_tokens": 0}

        total_latency = sum(c.latency_seconds for c in self.calls)
        total_completion_tokens = sum(c.completion_tokens for c in self.calls)
        total_prompt_tokens = sum(c.prompt_tokens for c in self.calls)

        avg_latency = total_latency / total_calls
        throughput = total_completion_tokens / total_latency if total_latency > 0 else 0.0

        return {
            "total_calls": total_calls,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "avg_latency_seconds": round(avg_latency, 4),
            "throughput_tokens_per_sec": round(throughput, 2),
        }