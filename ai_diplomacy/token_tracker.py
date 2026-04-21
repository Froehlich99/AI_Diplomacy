import asyncio
import json
import csv
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger("token_tracker")


@dataclass
class LLMCallRecord:
    timestamp: str
    model: str
    power: str
    phase: str
    response_type: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class TokenTracker:
    _instance: Optional["TokenTracker"] = None

    def __init__(self, preload_path: Optional[str] = None):
        self.records: list[LLMCallRecord] = []
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self._lock = asyncio.Lock()
        self._context_power: str = "unknown"
        self._context_phase: str = "unknown"
        self._context_response_type: str = "unknown"

        if preload_path:
            self._load_existing(preload_path)

    def _load_existing(self, filepath: str):
        """Load records from a previous token_usage.json so totals survive restarts."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("records", []):
                rec = LLMCallRecord(
                    timestamp=r["timestamp"],
                    model=r["model"],
                    power=r["power"],
                    phase=r["phase"],
                    response_type=r["response_type"],
                    input_tokens=r["input_tokens"],
                    output_tokens=r["output_tokens"],
                    cost_usd=r["cost_usd"],
                )
                self.records.append(rec)
                self.total_cost += rec.cost_usd
                self.total_input_tokens += rec.input_tokens
                self.total_output_tokens += rec.output_tokens
            if self.records:
                logger.info("Loaded %d existing records (%.4f USD) from %s", len(self.records), self.total_cost, filepath)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Could not load existing token usage from %s: %s", filepath, e)

    def set_context(self, power: Optional[str] = None, phase: Optional[str] = None, response_type: Optional[str] = None):
        if power is not None:
            self._context_power = power
        if phase is not None:
            self._context_phase = phase
        if response_type is not None:
            self._context_response_type = response_type

    async def record(self, model: str, input_tokens: int, output_tokens: int, cost: Optional[float] = None):
        rec = LLMCallRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            power=self._context_power,
            phase=self._context_phase,
            response_type=self._context_response_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost or 0.0,
        )
        async with self._lock:
            self.records.append(rec)
            self.total_cost += rec.cost_usd
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

    def get_summary(self) -> dict:
        by_model: dict[str, dict] = {}
        by_power: dict[str, dict] = {}
        by_phase: dict[str, dict] = {}

        for r in self.records:
            for key, bucket in ((r.model, by_model), (r.power, by_power), (r.phase, by_phase)):
                if key not in bucket:
                    bucket[key] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0}
                bucket[key]["input_tokens"] += r.input_tokens
                bucket[key]["output_tokens"] += r.output_tokens
                bucket[key]["cost_usd"] += r.cost_usd
                bucket[key]["calls"] += 1

        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": len(self.records),
            "by_model": by_model,
            "by_power": by_power,
            "by_phase": by_phase,
        }

    def export_json(self, filepath: str):
        summary = self.get_summary()
        summary["records"] = [asdict(r) for r in self.records]
        try:
            tmp = f"{filepath}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            os.rename(tmp, filepath)
        except Exception as e:
            logger.error("Failed to export token usage to %s: %s", filepath, e)

    def export_csv(self, filepath: str):
        if not self.records:
            return
        fieldnames = ["timestamp", "model", "power", "phase", "response_type", "input_tokens", "output_tokens", "cost_usd"]
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.records:
                    writer.writerow(asdict(r))
        except Exception as e:
            logger.error("Failed to export token CSV to %s: %s", filepath, e)


def init_tracker(preload_path: Optional[str] = None) -> TokenTracker:
    TokenTracker._instance = TokenTracker(preload_path=preload_path)
    return TokenTracker._instance


def get_tracker() -> TokenTracker:
    if TokenTracker._instance is None:
        TokenTracker._instance = TokenTracker()
    return TokenTracker._instance
