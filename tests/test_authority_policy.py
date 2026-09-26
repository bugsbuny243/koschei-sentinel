from datetime import datetime, timedelta, timezone

import pytest

from koschei_sentinel.authority_policy import (
    AuthorityPolicy,
    AuthorityPolicyError,
    CapabilityGrant,
    DelegationChain,
)
from koschei_sentinel.inference_contract import AuthorityEnvelope


NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def grant(gid, issuer, subject, scopes, **kwargs):
    return CapabilityGrant(gid, issuer, subject, tuple(scopes), **kwargs)


def authority(delegate="agent-b", scopes=("read", "analyze")):
    return AuthorityEnvelope(
        principal_id="user-a",
        controller_id="controller-a",
        delegate_id=delegate,
        scopes=tuple(scopes),
    )


def test_attenuated_chain_authorizes_subset():
    chain = DelegationChain((
        grant("g1", "user-a", "agent-a", ("read", "analyze", "write")),
        grant("g2", "agent-a", "agent-b", ("read", "analyze")),
    ))
    decision = AuthorityPolicy().evaluate(
        authority(), chain, required_scopes=("analyze",), now=NOW
    )
    assert decision.allowed
    assert decision.effective_scopes == ("analyze", "read")


def test_child_cannot_widen_parent_authority():
    chain = DelegationChain((
        grant("g1", "user-a", "agent-a", ("read",)),
        grant("g2", "agent-a", "agent-b", ("read", "write")),
    ))
    with pytest.raises(AuthorityPolicyError, match="widened"):
        AuthorityPolicy().evaluate(authority(scopes=("read", "write")), chain, required_scopes=("read",), now=NOW)


def test_no_further_delegation_is_enforced():
    chain = DelegationChain((
        grant("g1", "user-a", "agent-a", ("read",), no_further_delegation=True),
        grant("g2", "agent-a", "agent-b", ("read",)),
    ))
    with pytest.raises(AuthorityPolicyError, match="forbids"):
        AuthorityPolicy().evaluate(authority(scopes=("read",)), chain, required_scopes=("read",), now=NOW)


def test_expired_grant_fails_closed():
    chain = DelegationChain((
        grant("g1", "user-a", "agent-b", ("read",), expires_at=NOW - timedelta(seconds=1)),
    ))
    with pytest.raises(AuthorityPolicyError, match="expired"):
        AuthorityPolicy().evaluate(authority(scopes=("read",)), chain, required_scopes=("read",), now=NOW)


def test_missing_required_scope_denies_without_expanding_authority():
    chain = DelegationChain((
        grant("g1", "user-a", "agent-b", ("read",)),
    ))
    decision = AuthorityPolicy().evaluate(
        authority(scopes=("read",)), chain, required_scopes=("write",), now=NOW
    )
    assert not decision.allowed
    assert decision.effective_scopes == ("read",)
