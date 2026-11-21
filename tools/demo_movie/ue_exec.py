"""Run a Python file inside the running Unreal editor (Python Remote Execution).

The MCP ``ProgrammaticToolset`` is a sandbox without ``import unreal``, so the film tools talk to the
editor's Python plugin directly, with the engine's own client
(``Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py``).

Remote execution must be on in the editor. Without touching the tracked config, launch the editor with
``-ini:Engine:[/Script/PythonScriptPlugin.PythonScriptPluginSettings]:bRemoteExecution=True``
(``film.py editor`` does), or tick Project Settings > Plugins > Python > Enable Remote Execution.

Scripts get their arguments as JSON in ``sys.argv[1]``; whatever they print comes back as the output.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ENGINE_DIR = Path(os.environ.get("UE58_DIR", r"C:\Program Files\Epic Games\UE_5.8"))
REMOTE_EXECUTION_PY = ENGINE_DIR / "Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py"
REMOTE_EXECUTION_INI = "-ini:Engine:[/Script/PythonScriptPlugin.PythonScriptPluginSettings]:bRemoteExecution=True"
ROOT = Path(__file__).resolve().parents[2]
ARGS_DIR = ROOT / "runs" / "_film" / "_args"


class EditorPythonError(RuntimeError):
    pass


def _module():
    spec = importlib.util.spec_from_file_location("ue_remote_execution", REMOTE_EXECUTION_PY)
    if spec is None or not REMOTE_EXECUTION_PY.exists():
        raise EditorPythonError(f"remote_execution.py not found at {REMOTE_EXECUTION_PY} (set UE58_DIR)")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EditorPython:
    """A command connection to the one running editor that has remote execution on."""

    def __init__(self, project_name: str = "VIVID2Env", discover_s: float = 8.0):
        self._re = _module()
        self.remote = self._re.RemoteExecution()
        self.remote.start()
        deadline = time.monotonic() + discover_s
        node = None
        while time.monotonic() < deadline:
            nodes = [n for n in self.remote.remote_nodes if n.get("project_name") == project_name]
            if nodes:
                node = nodes[0]
                break
            time.sleep(0.25)
        if node is None:
            self.remote.stop()
            raise EditorPythonError(
                f"no {project_name} editor with Python remote execution found (start it with 'film.py editor')")
        self.node = node
        self.remote.open_command_connection(node["node_id"])

    def close(self) -> None:
        try:
            self.remote.stop()
        except Exception:
            pass

    def __enter__(self) -> "EditorPython":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run_file(self, path: Path, args: Optional[Dict[str, Any]] = None, raise_on_failure: bool = True) -> str:
        """Execute a .py file in the editor; ``args`` reach it as a JSON file (``script_args()``).

        Returns the script's printed output."""
        path = Path(path).resolve()
        ARGS_DIR.mkdir(parents=True, exist_ok=True)
        args_path = ARGS_DIR / f"{path.stem}_{os.getpid()}_{int(time.time() * 1000)}.json"
        args_path.write_text(json.dumps(args or {}), encoding="utf-8")
        if " " in path.as_posix() or " " in args_path.as_posix():
            raise EditorPythonError(f"paths with spaces can't be passed to the editor: {path} {args_path}")
        cmd = f"{path.as_posix()} {args_path.as_posix()}"
        res = self.remote.run_command(cmd, unattended=True, exec_mode=self._re.MODE_EXEC_FILE)
        out = "\n".join(o.get("output", "") for o in res.get("output", []))
        if raise_on_failure and not res.get("success", False):
            raise EditorPythonError(f"{path.name} failed:\n{out}\n{res.get('result', '')}")
        return out

    def eval(self, statement: str) -> str:
        res = self.remote.run_command(statement, unattended=True, exec_mode=self._re.MODE_EVAL_STATEMENT)
        if not res.get("success", False):
            raise EditorPythonError(f"{statement}: {res.get('result')}")
        return str(res.get("result"))


def script_args() -> Dict[str, Any]:
    """Inside an editor script: the args passed by ``run_file``."""
    if len(sys.argv) < 2:
        return {}
    return json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))


def last_json(output: str) -> Any:
    """The last line of an editor script's output that parses as JSON (scripts print a result line)."""
    for line in reversed([l for l in output.splitlines() if l.strip()]):
        try:
            return json.loads(line)
        except ValueError:
            continue
    raise EditorPythonError(f"no JSON result in output:\n{output[-2000:]}")


def editor_command(project: Path, level: Optional[str] = None, extra: Optional[List[str]] = None) -> List[str]:
    exe = ENGINE_DIR / "Engine/Binaries/Win64/UnrealEditor.exe"
    cmd = [str(exe), str(project)]
    if level:
        cmd.append(level)
    cmd += ['-ExecCmds="ModelContextProtocol.StartServer"', REMOTE_EXECUTION_INI]
    return cmd + list(extra or [])
