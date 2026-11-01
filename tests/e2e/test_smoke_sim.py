"""Smoke campaign against a running Project AirSim server (Blocks or VIVID2Env).

    pytest -m sim tests/e2e
"""
import asyncio

import pytest

from vivid2.registry import AgentFactory
from vivid2.runner import execute_campaign
from vivid2.scenario import Campaign

pytestmark = pytest.mark.sim


def test_oracle_stack_on_blocks(root, tmp_path):
    pytest.importorskip("projectairsim")
    campaign = Campaign.load(root / "campaigns/smoke.jsonc")
    campaign.seeds = [0]
    results = asyncio.run(execute_campaign(campaign, AgentFactory("coverage_gt"), tmp_path, resume=False))
    assert len(results) == 1
    r = results[0]
    assert r["metrics"]["mission"]["end_reason"] in ("completed", "timeout")
    assert r["metrics"]["planning"]["coverage_pct"] > 50
    assert r["metrics"]["mission"]["success"], r["metrics"]
