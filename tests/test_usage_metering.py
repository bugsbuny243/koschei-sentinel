from decimal import Decimal

import pytest

from koschei_sentinel.usage_metering import (
    UsageBudget,
    UsageEvent,
    UsageLimitExceeded,
    UsageRate,
    calculate_charge,
    enforce_budget,
)


def test_charge_is_deterministic():
    event = UsageEvent("r1", "t1", 2000, 500, 2)
    rate = UsageRate(Decimal("0.10"), Decimal("0.40"), Decimal("0.25"))
    assert calculate_charge(event, rate).credits == Decimal("0.90")


def test_budget_accepts_charge_within_limit():
    budget = UsageBudget("t1", Decimal(10))
    charge = calculate_charge(
        UsageEvent("r1", "t1", 1000, 1000),
        UsageRate(Decimal(1), Decimal(2)),
    )
    assert enforce_budget(budget=budget, consumed=Decimal(5), charge=charge) == Decimal(8)


def test_budget_fails_closed_when_limit_exceeded():
    budget = UsageBudget("t1", Decimal(5))
    charge = calculate_charge(
        UsageEvent("r1", "t1", 1000, 1000),
        UsageRate(Decimal(1), Decimal(2)),
    )
    with pytest.raises(UsageLimitExceeded):
        enforce_budget(budget=budget, consumed=Decimal(4), charge=charge)


def test_cross_tenant_charge_is_rejected():
    budget = UsageBudget("t1", Decimal(10))
    charge = calculate_charge(
        UsageEvent("r1", "t2", 1000, 0), UsageRate(Decimal(1), Decimal(0))
    )
    with pytest.raises(ValueError, match="tenant mismatch"):
        enforce_budget(budget=budget, consumed=Decimal(0), charge=charge)
