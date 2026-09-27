from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


class UsageLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class UsageBudget:
    tenant_id: str
    credit_limit: Decimal


@dataclass(frozen=True)
class UsageEvent:
    request_id: str
    tenant_id: str
    input_units: int
    output_units: int
    tool_units: int = 0

    def __post_init__(self) -> None:
        if min(self.input_units, self.output_units, self.tool_units) < 0:
            raise ValueError("usage units cannot be negative")


@dataclass(frozen=True)
class UsageRate:
    input_per_1000: Decimal
    output_per_1000: Decimal
    tool_per_unit: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if min(self.input_per_1000, self.output_per_1000, self.tool_per_unit) < 0:
            raise ValueError("usage rates cannot be negative")


@dataclass(frozen=True)
class UsageCharge:
    request_id: str
    tenant_id: str
    credits: Decimal


def calculate_charge(event: UsageEvent, rate: UsageRate) -> UsageCharge:
    credits = (
        Decimal(event.input_units) * rate.input_per_1000 / Decimal(1000)
        + Decimal(event.output_units) * rate.output_per_1000 / Decimal(1000)
        + Decimal(event.tool_units) * rate.tool_per_unit
    )
    return UsageCharge(event.request_id, event.tenant_id, credits)


def enforce_budget(*, budget: UsageBudget, consumed: Decimal, charge: UsageCharge) -> Decimal:
    if charge.tenant_id != budget.tenant_id:
        raise ValueError("usage charge tenant mismatch")
    if consumed < 0:
        raise ValueError("consumed credits cannot be negative")
    projected = consumed + charge.credits
    if projected > budget.credit_limit:
        raise UsageLimitExceeded("tenant credit limit exceeded")
    return projected
