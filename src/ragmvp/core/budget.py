"""Token budgets: cap request size, per-user daily usage and the global monthly budget.

A limit of 0 means "unlimited" (e.g. a local Ollama model); set limits for commercial models.
Token counts before the call are estimates (about 4 characters per token); recorded usage
uses the provider's numbers when available.
"""
from __future__ import annotations

from dataclasses import dataclass

from ragmvp.core.models import Usage
from ragmvp.core.ports import OpsPort


class BudgetExceeded(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class BudgetLimits:
    max_output_tokens: int = 700
    max_request_tokens: int = 6000
    daily_tokens_per_user: int = 0
    monthly_token_budget: int = 0


class TokenBudget:
    def __init__(self, ops: OpsPort, limits: BudgetLimits):
        self._ops = ops
        self.limits = limits

    def check(self, user_id: str, estimated_input_tokens: int) -> int:
        """Raise BudgetExceeded if the call is not allowed; return the max output tokens to request."""
        lim = self.limits
        if lim.max_request_tokens and estimated_input_tokens > lim.max_request_tokens:
            raise BudgetExceeded("request_too_large")
        needed = estimated_input_tokens + lim.max_output_tokens
        if lim.daily_tokens_per_user and self._ops.tokens_used_today(user_id) + needed > lim.daily_tokens_per_user:
            raise BudgetExceeded("daily_user_limit")
        if lim.monthly_token_budget and self._ops.tokens_used_this_month() + needed > lim.monthly_token_budget:
            raise BudgetExceeded("monthly_budget")
        return lim.max_output_tokens

    def record(self, user_id: str, model: str, usage: Usage) -> None:
        self._ops.record_usage(user_id, model, usage)
