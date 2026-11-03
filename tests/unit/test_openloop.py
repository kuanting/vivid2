"""Frame bank -> open-loop evaluation -> record/replay must reproduce scores exactly."""
import asyncio
import json

import numpy as np
from PIL import Image

from conftest import FakeWorld
from vivid2.openloop import evaluate_run, evaluate_runs, find_banked_runs
from vivid2.recorder import Recorder
from vivid2.report import write_report
from vivid2.runner import mission_spec, write_bank_files
from vivid2.stack.base import UsageStats
from vivid2.stack.reference.agents import SurveyAgent
from vivid2.stack.reference.planners import LawnmowerPlanner
from vivid2.stack.reference.reasoners import VLMReasoner
from vivid2.vlm.base import VLMProvider, decode_jpeg, encode_jpeg_bytes, image_bytes
from vivid2.vlm.openai_compat import pixels_to_norm1000
from vivid2.vlm.replay import ReplayVLM


class BrightnessVLM(VLMProvider):
    """Deterministic stand-in: 'finds' a target when the image is bright."""

    model = "brightness-vlm"

    def __init__(self):
        self.calls = 0

    def query(self, images, prompt, schema, system=None):
        self.calls += 1
        bright = float(decode_jpeg(image_bytes(images[0])).mean()) > 100
        return ({"target_found": bright, "confidence": 0.9 if bright else 0.1, "point": [500, 500] if bright else [-1, -1],
                 "description": "bright" if bright else "dark", "hazards": [], "next_action": "continue"},
                UsageStats(latency_s=1.5, input_tokens=1200, output_tokens=80, cost_usd=0.01, model=self.model))


def _banked_run(tmp_path, blocks_run, n=6):
    run_dir = tmp_path / "runs" / "blocks_v1" / "survey" / blocks_run.run_id
    run_dir.mkdir(parents=True)
    frames = []
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        level = 200 if i % 3 == 0 else 30  # every third frame is "bright" and has a visible target
        img = np.clip(rng.normal(level, 5, (72, 96, 3)), 0, 255).astype(np.uint8)
        bank = write_bank_files(run_dir, i, encode_jpeg_bytes(img), None)
        gt = [{"object_id": "Hiker1", "bbox": [40, 30, 56, 42]}] if level == 200 else []
        frames.append({"t": i * 10**9, "frame_id": i, "cam_pos": [0, 0, -35], "cam_q": [1, 0, 0, 0],
                       "width": 96, "height": 72, "fov_deg": 80, "gt": gt, **bank})
    (run_dir / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames), encoding="utf-8")
    meta = {"run_id": blocks_run.run_id, "scenario_id": blocks_run.scenario_id, "seed": 0, "params": {},
            "mission": "forest_sar", "agent": {"agent": "survey"}, "targets": blocks_run.resolved_targets(),
            "mission_spec": mission_spec(blocks_run).record(), "frame_bank": {"mode": "rgb"}}
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


def test_bank_files_round_trip(tmp_path):
    jpeg = encode_jpeg_bytes(np.full((8, 8, 3), 120, np.uint8))
    depth = np.full((8, 8), 33.25, np.float32)
    rec = write_bank_files(tmp_path, 7, jpeg, depth)
    assert (tmp_path / rec["rgb"]).read_bytes() == jpeg
    mm = np.asarray(Image.open(tmp_path / rec["depth"]))
    assert mm.shape == (4, 4) and int(mm[0, 0]) == 33250


def test_open_loop_record_then_replay_is_exact(tmp_path, blocks_run):
    run_dir = _banked_run(tmp_path, blocks_run)
    assert find_banked_runs([tmp_path / "runs"]) == [run_dir]
    cache = tmp_path / "cache.jsonl"
    live = BrightnessVLM()
    recorded = evaluate_runs([run_dir], VLMReasoner(ReplayVLM(str(cache), live, "record")), "vlm-brightness",
                             {"agent": "vlm-brightness"}, log=lambda _: None)[0]
    assert live.calls == 6
    r = recorded["metrics"]["reasoning"]
    assert r["confusion"] == {"tp": 2, "fp": 0, "tn": 4, "fn": 0} and r["errors"] == 0

    replayer = BrightnessVLM()
    replayed = evaluate_runs([run_dir], VLMReasoner(ReplayVLM(str(cache), replayer, "replay")), "vlm-brightness-replay",
                             {"agent": "vlm-brightness-replay"}, log=lambda _: None)[0]
    assert replayer.calls == 0
    assert replayed["metrics"] == recorded["metrics"] and replayed["scores"] == recorded["scores"]


def test_open_loop_resumes_per_frame(tmp_path, blocks_run):
    run_dir = _banked_run(tmp_path, blocks_run)
    vlm = BrightnessVLM()
    out_root = tmp_path / "runs" / "blocks_v1" / "openloop"
    evaluate_run(run_dir, VLMReasoner(vlm), out_root, "vlm-b", {"agent": "vlm-b"}, log=lambda _: None)
    evaluate_run(run_dir, VLMReasoner(vlm), out_root, "vlm-b", {"agent": "vlm-b"}, log=lambda _: None)
    assert vlm.calls == 6  # the second pass had nothing left to do


def test_report_lists_open_loop_models_separately(tmp_path, blocks_run):
    run_dir = _banked_run(tmp_path, blocks_run)
    evaluate_runs([run_dir], VLMReasoner(BrightnessVLM()), "vlm-brightness", {"agent": "vlm-brightness"}, log=lambda _: None)
    html = write_report(tmp_path / "runs" / "blocks_v1").read_text(encoding="utf-8")
    assert "Open-loop VLM comparison" in html and "vlm-brightness" in html


def test_pixel_points_are_rescaled_for_grounding():
    out = pixels_to_norm1000({"target_found": True, "point": [480, 180]}, (960, 720))
    assert out["point"] == [500.0, 250.0] and out["point_raw"] == [480, 180]
    assert pixels_to_norm1000({"point": [-1, -1]}, (960, 720)) == {"point": [-1, -1]}


def test_survey_agent_records_frames_without_decisions(tmp_path, blocks_run):
    rec = Recorder(tmp_path / "survey")
    world = FakeWorld(blocks_run, rec)
    asyncio.run(SurveyAgent("survey", LawnmowerPlanner(), frame_period_s=0.0).run(world))
    rec.close()
    assert world.frame_id > 10
    assert (tmp_path / "survey" / "decisions.jsonl").read_text(encoding="utf-8") == ""
    assert np.linalg.norm(world.pos[:2] - np.asarray(blocks_run.spec["dut_start"]["ned"][:2])) < 1.0
