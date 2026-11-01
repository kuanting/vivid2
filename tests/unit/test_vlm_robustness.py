"""VLM failures must be scored as abstentions and never crash a run."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vivid2.metrics.reasoning import reasoning_metrics
from vivid2.recorder import RunLog
from vivid2.registry import AgentFactory
from vivid2.runner import mission_spec
from vivid2.scoring import layer_scores
from vivid2.stack.base import Frame, UsageStats
from vivid2.stack.reference.reasoners import ASSESSMENT_SCHEMA, VLMReasoner, parse_assessment
from vivid2.vlm.base import VLMError, VLMProvider
from vivid2.vlm.openai_compat import _extract_json

NO_TARGET = {"target_found": False}
ERROR = {"target_found": False, "error": "API error 529: overloaded"}
IMG = np.zeros((8, 8, 3), np.uint8)


class ReplyVLM(VLMProvider):
    model = "reply-vlm"

    def __init__(self, reply):
        self.reply = reply

    def query(self, images, prompt, schema, system=None):
        return self.reply, UsageStats(input_tokens=100, model=self.model)


def _decision_log(outputs, visible):
    frames = [{"frame_id": i, "gt": [{"object_id": "A", "bbox": [0, 0, 10, 10]}] if v else []}
              for i, v in enumerate(visible)]
    decisions = [{"frame_id": i, "layer": "reasoner", "output": o, "latency_s": 1.0} for i, o in enumerate(outputs)]
    return RunLog(run_dir=Path("."), meta={}, streams={"frames": frames, "decisions": decisions})


def _reasoning_score(r):
    m = {"mission": {"timeout_s": 100, "time_to_find_s": {}, "targets_total": 1, "false_alarms": 0},
         "perception": None, "reasoning": r, "planning": {"coverage_pct": 0.0}, "control": {}, "safety": {"collisions": 0}}
    return layer_scores(m)["reasoning"]


def test_vlm_errors_are_abstentions_not_negatives():
    empty = [False] * 4
    good = reasoning_metrics(_decision_log([NO_TARGET] * 4, empty))
    flaky = reasoning_metrics(_decision_log([NO_TARGET, ERROR, ERROR, ERROR], empty))
    assert flaky["errors"] == 3 and flaky["error_rate"] == pytest.approx(0.75)
    assert flaky["confusion"] == {"tp": 0, "fp": 0, "tn": 1, "fn": 0}
    assert flaky["accuracy"] == 1.0  # over the one decision it actually made
    # Erroring must cost score, not look like correct "no target" answers.
    assert _reasoning_score(flaky) == pytest.approx(0.25 * _reasoning_score(good))
    all_err = reasoning_metrics(_decision_log([ERROR, ERROR], [True, False]))
    assert all_err["accuracy"] is None and _reasoning_score(all_err) == 0.0


def test_parse_assessment_coerces_loose_fields():
    a = parse_assessment({"target_found": "false", "confidence": None, "point": None}, 800, 600)
    assert a.target_found is False and a.confidence == 0.0 and a.pixel is None
    a = parse_assessment({"target_found": "true", "confidence": "0.9", "point": [500, "x"],
                          "hazards": None, "next_action": "dance"}, 800, 600)
    assert a.target_found and a.confidence == 0.9 and a.pixel is None
    assert a.hazards == [] and a.next_action == "continue"
    with pytest.raises(ValueError):
        parse_assessment({"confidence": 1.0}, 800, 600)
    with pytest.raises(ValueError):
        parse_assessment({"target_found": "maybe"}, 800, 600)


def test_reasoner_turns_unusable_reply_into_error(blocks_run):
    frame = Frame(0, 1, np.zeros((600, 800, 3), np.uint8), np.zeros(3), (1, 0, 0, 0), 80.0)
    a, usage = VLMReasoner(ReplyVLM({"target_found": None})).assess(frame, [], mission_spec(blocks_run))
    assert not a.target_found and "error" in a.raw
    assert usage.input_tokens == 100  # the call still cost tokens


def test_local_vlm_json_extraction_raises_vlm_error():
    assert _extract_json('noise {"target_found": true} noise') == {"target_found": True}
    for bad in ["no json here", "{broken: json}", "[1, 2]"]:
        with pytest.raises(VLMError):
            _extract_json(bad)


def _fake_response(text, model="claude-opus-5-5", fallback=False):
    iterations = [SimpleNamespace(type="message"), SimpleNamespace(type="fallback_message")] if fallback else None
    return SimpleNamespace(model=model, stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000, iterations=iterations))


def _anthropic_vlm(monkeypatch, resp):
    pytest.importorskip("anthropic")
    from vivid2.vlm.anthropic_provider import AnthropicVLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    vlm = AnthropicVLM()

    def create(**kwargs):
        return resp

    vlm.client = SimpleNamespace(messages=SimpleNamespace(create=create),
                                 beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    return vlm


def test_anthropic_defaults_and_prices_the_serving_model(monkeypatch):
    reply = json.dumps({"target_found": False})
    vlm = _anthropic_vlm(monkeypatch, _fake_response(reply))
    assert vlm.model == "claude-opus-5-5" and vlm.max_tokens >= 16000
    out, usage = vlm.query([IMG], "p", ASSESSMENT_SCHEMA)
    assert out == {"target_found": False} and not usage.fallback
    assert usage.cost_usd == pytest.approx(4.0 + 2.0)  # 1M in at $4/M + 0.1M out at $20/M

    vlm = _anthropic_vlm(monkeypatch, _fake_response(reply, model="claude-sonnet-5-5", fallback=True))
    _, usage = vlm.query([IMG], "p", ASSESSMENT_SCHEMA)
    assert usage.model == "claude-sonnet-5-5" and usage.fallback
    assert usage.cost_usd == pytest.approx(2.0 + 1.0)  # priced at the serving model's rates


def test_anthropic_bad_json_is_vlm_error(monkeypatch):
    vlm = _anthropic_vlm(monkeypatch, _fake_response("not json"))
    with pytest.raises(VLMError):
        vlm.query([IMG], "p", ASSESSMENT_SCHEMA)


def test_replay_without_cache_is_a_clear_error():
    with pytest.raises(ValueError, match="--vlm-cache"):
        AgentFactory("vlm", {"vlm": "replay"})
