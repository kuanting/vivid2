"""Editor side: render shot sequences with Movie Render Queue (PIE executor) to PNG sequences.

Runs inside the Unreal editor via ``film.py ue-render``. Rendering continues after this script
returns; progress and the end are written to ``<out>/render_status.json``, which ``film.py`` polls.

Each shot is one job: <out>/<shot>/<sequence>.<frame 6 digits>.png. The game mode is overridden with
GameModeBase, so Project AirSim's game mode (which would launch the simulation) never runs.
"""
import builtins
import json
import sys
import time
from pathlib import Path

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ue_exec import script_args  # noqa: E402

ARGS = script_args()
OUT = Path(ARGS["out"])
STATUS = OUT / "render_status.json"
PRESETS = {
    # quick look: one sample, TSR
    "preview": {"res": [960, 540], "spatial": 1, "temporal": 1, "aa": "tsr", "engine_warm_up": 16, "render_warm_up": 8},
    # final: 8 temporal samples give motion blur on propellers and camera moves
    "final": {"res": [1920, 1080], "spatial": 1, "temporal": 8, "aa": "none", "engine_warm_up": 32, "render_warm_up": 64},
}


def write_status(**kw):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    state.update(kw, t=time.time())
    STATUS.write_text(json.dumps(state))


def configure(job, preset, out_dir):
    cfg = job.get_configuration()
    out = cfg.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    out.set_editor_property("output_directory", unreal.DirectoryPath(str(out_dir).replace("\\", "/")))
    out.set_editor_property("output_resolution", unreal.IntPoint(*preset["res"]))
    out.set_editor_property("file_name_format", out_dir.name + ".{frame_number}")
    out.set_editor_property("zero_pad_frame_numbers", 6)
    out.set_editor_property("override_existing_output", True)
    cfg.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    cfg.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    aa = cfg.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.set_editor_property("spatial_sample_count", preset["spatial"])
    aa.set_editor_property("temporal_sample_count", preset["temporal"])
    aa.set_editor_property("override_anti_aliasing", True)
    aa.set_editor_property("anti_aliasing_method", unreal.AntiAliasingMethod.AAM_TSR if preset["aa"] == "tsr"
                           else unreal.AntiAliasingMethod.AAM_NONE)
    aa.set_editor_property("engine_warm_up_count", preset["engine_warm_up"])
    aa.set_editor_property("render_warm_up_frames", True)
    aa.set_editor_property("render_warm_up_count", preset["render_warm_up"])
    go = cfg.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
    go.set_editor_property("game_mode_override", unreal.GameModeBase.static_class())
    go.set_editor_property("cinematic_quality_settings", True)
    go.set_editor_property("use_lod_zero", True)
    go.set_editor_property("texture_streaming", unreal.MoviePipelineTextureStreamingMethod.FULLY_LOAD)
    if ARGS.get("cvars"):
        cv = cfg.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
        for name, value in ARGS["cvars"].items():
            cv.add_or_update_console_variable(name, float(value))


def main():
    subsys = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    if subsys.is_rendering():
        raise RuntimeError("Movie Render Queue is already rendering")
    preset = dict(PRESETS[ARGS.get("preset", "preview")])
    if ARGS.get("res"):
        preset["res"] = ARGS["res"]
    queue = subsys.get_queue()
    queue.delete_all_jobs()
    for shot in ARGS["shots"]:
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.set_editor_property("job_name", shot["id"])
        job.set_editor_property("sequence", unreal.SoftObjectPath(shot["sequence"]))
        job.set_editor_property("map", unreal.SoftObjectPath(ARGS["map"]))
        configure(job, preset, OUT / shot["id"])
    if STATUS.exists():
        STATUS.unlink()
    write_status(state="rendering", shots=[s["id"] for s in ARGS["shots"]], done=[], preset=ARGS.get("preset"))

    executor = unreal.MoviePipelinePIEExecutor()

    def on_job(job, success):
        st = json.loads(STATUS.read_text())
        write_status(done=st.get("done", []) + [{"id": str(job.get_editor_property("job_name")), "ok": bool(success)}])

    def on_finished(exe, success):
        write_status(state="finished", ok=bool(success))

    if hasattr(executor, "on_individual_job_finished_delegate"):
        executor.on_individual_job_finished_delegate.add_callable(on_job)
    executor.on_executor_finished_delegate.add_callable(on_finished)
    builtins._vivid2_film_executor = executor  # keep it alive after this script returns
    subsys.render_queue_with_executor_instance(executor)
    print(json.dumps({"queued": [s["id"] for s in ARGS["shots"]], "status": str(STATUS)}))


main()
