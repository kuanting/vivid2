"""Set up unreal/VIVID2Env for a chosen Unreal Engine version (5.7 or 5.8).

The Project AirSim plugin source compiles for both engines once the 5.8
compatibility patch is applied; only built binaries are engine-specific. A
user can get a working plugin in one of four ways:

    --plugin-dir   an already-built plugin folder (e.g. ..\\PASBlocks\\Plugins)
    --airsim-root  a Project AirSim checkout (5.8 needs patches/projectairsim-ue5.8.patch)
    --plugin-zip   a prebuilt plugin zip downloaded manually
    --download     the prebuilt plugin zip listed in unreal/engines.json

The plugin folders are linked (directory junctions on Windows) into
unreal/VIVID2Env/Plugins, VIVID2Env.uproject is rendered for the engine, and
the result is recorded in unreal/VIVID2Env/.vivid2-setup.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
UNREAL = REPO / "unreal"
ENV_DIR = UNREAL / "VIVID2Env"
PLUGIN_NAMES = ("ProjectAirSim", "Drone")
EXCLUDED_DIRS = {"Intermediate", ".git"}
STATE_FILE = ".vivid2-setup.json"
FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class SetupError(RuntimeError):
    pass


# -- engine profiles / uproject -------------------------------------------

def load_engines(path: Optional[Path] = None) -> Dict[str, Any]:
    return json.loads(Path(path or UNREAL / "engines.json").read_text(encoding="utf-8"))


def engine_profile(engine: str, engines: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    engines = engines or load_engines()
    try:
        return engines["engines"][engine]
    except KeyError:
        raise SetupError(f"unsupported engine {engine!r}; supported: {sorted(engines['engines'])}") from None


def render_uproject(template: Dict[str, Any], engine: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(template))
    out["EngineAssociation"] = engine
    names = {p["Name"] for p in out.get("Plugins", [])}
    for name in profile.get("extra_plugins", []):
        if name not in names:
            out.setdefault("Plugins", []).append({"Name": name, "Enabled": True})
    return out


def write_uproject(engine: str, env_dir: Path = ENV_DIR, engines: Optional[Dict[str, Any]] = None) -> Path:
    template = json.loads((env_dir / "VIVID2Env.uproject.template.json").read_text(encoding="utf-8"))
    path = env_dir / "VIVID2Env.uproject"
    path.write_text(json.dumps(render_uproject(template, engine, engine_profile(engine, engines)), indent="\t") + "\n",
                    encoding="utf-8")
    return path


# -- engine / plugin inspection ---------------------------------------------

def find_engine_dir(engine: str) -> Optional[Path]:
    """Locate an installed launcher engine via the registry, then the default path."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\EpicGames\Unreal Engine\{engine}") as k:
                p = Path(winreg.QueryValueEx(k, "InstalledDirectory")[0])
                if p.exists():
                    return p
        except OSError:
            pass
    p = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Epic Games" / f"UE_{engine}"
    return p if p.exists() else None


def read_build_id(modules_file: Path) -> Optional[str]:
    try:
        return str(json.loads(modules_file.read_text(encoding="utf-8")).get("BuildId"))
    except (OSError, ValueError):
        return None


def engine_build_id(engine_dir: Optional[Path]) -> Optional[str]:
    return read_build_id(engine_dir / "Engine/Binaries/Win64/UnrealEditor.modules") if engine_dir else None


def plugin_build_id(plugins_dir: Path) -> Optional[str]:
    return read_build_id(plugins_dir / "ProjectAirSim/Binaries/Win64/UnrealEditor.modules")


def simlibs_commit(plugins_dir: Path) -> Optional[str]:
    p = plugins_dir / "ProjectAirSim/SimLibs/commit_hash.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def supports_ue58(airsim_root: Path) -> bool:
    header = airsim_root / "unreal/Blocks/Plugins/ProjectAirSim/Source/ProjectAirSim/Private/UnrealCompatibility.h"
    return header.exists() and "UE_IS_5_8" in header.read_text(encoding="utf-8", errors="ignore")


# Teams of drones: each drone's cameras must see only its own weather (fog) actor, any engine.
TEAM_WEATHER_PATCH = "patches/projectairsim-team-weather.patch"


def has_team_weather_fix(airsim_root: Path) -> bool:
    src = airsim_root / "unreal/Blocks/Plugins/ProjectAirSim/Source/ProjectAirSim/Private/World/WeatherLib.cpp"
    return src.exists() and "HideOtherRobotsWeather" in src.read_text(encoding="utf-8", errors="ignore")


# -- linking -------------------------------------------------------------------

def _is_link(p: Path) -> bool:
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    return os.path.islink(p) or bool(getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT)


def link_dir(target: Path, link: Path, force: bool = False) -> None:
    """Point ``link`` at ``target``: a junction on Windows, a symlink elsewhere."""
    if _is_link(link):
        os.rmdir(link) if sys.platform == "win32" else link.unlink()
    elif link.exists():
        if not force:
            raise SetupError(f"{link} is a real directory; move it away or pass --force")
        shutil.rmtree(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def link_plugins(plugins_dir: Path, env_dir: Path = ENV_DIR, force: bool = False) -> None:
    for name in PLUGIN_NAMES:
        src = plugins_dir / name
        if not src.is_dir():
            raise SetupError(f"missing plugin folder {src}")
        link_dir(src.resolve(), env_dir / "Plugins" / name, force)


# -- zip packaging / download --------------------------------------------------

def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_plugin_files(plugins_dir: Path, strip_pdb: bool = False) -> Iterator[Tuple[Path, str]]:
    for name in PLUGIN_NAMES:
        root = plugins_dir / name
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
            for fn in filenames:
                if strip_pdb and fn.lower().endswith(".pdb"):
                    continue
                p = Path(dirpath) / fn
                yield p, p.relative_to(plugins_dir).as_posix()


def pack_plugin(plugins_dir: Path, engine: str, out_dir: Path, strip_pdb: bool = False,
                progress: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    plugins_dir, out_dir = Path(plugins_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"ProjectAirSim-Plugins-UE{engine}-Win64.zip"
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=6) as zf:
        for path, arc in iter_plugin_files(plugins_dir, strip_pdb):
            zf.write(path, arc)
            n += 1
            if n % 500 == 0:
                progress(f"{n} files...")
        zf.writestr("VIVID2_PLUGIN.json", json.dumps({
            "engine": engine, "build_id": plugin_build_id(plugins_dir), "simlibs_commit": simlibs_commit(plugins_dir),
            "created": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))
    sha = sha256_file(zip_path)
    (out_dir / (zip_path.name + ".sha256")).write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")
    return {"zip": str(zip_path), "files": n, "sha256": sha, "size": zip_path.stat().st_size, "file_name": zip_path.name}


def cache_root() -> Path:
    base = os.environ.get("VIVID2_CACHE") or os.path.join(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"), "VIVID2")
    return Path(base) / "plugins"


def extract_plugin_zip(zip_path: Path, engine: str, expected_sha256: Optional[str] = None,
                       cache: Optional[Path] = None) -> Path:
    """Verify and extract a plugin zip into the cache; returns the Plugins dir."""
    sha = sha256_file(zip_path)
    if expected_sha256 and sha.lower() != expected_sha256.lower():
        raise SetupError(f"SHA-256 mismatch for {zip_path}: got {sha}, expected {expected_sha256}")
    dest = (cache or cache_root()) / f"ue{engine}" / sha[:8] / "Plugins"
    if (dest / "ProjectAirSim" / "ProjectAirSim.uplugin").exists():
        return dest
    tmp = dest.with_name("Plugins.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():  # refuse path traversal
            if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                raise SetupError(f"unsafe path in zip: {member}")
        zf.extractall(tmp)
    if not (tmp / "ProjectAirSim" / "ProjectAirSim.uplugin").exists():
        raise SetupError("zip does not contain ProjectAirSim/ProjectAirSim.uplugin at its root")
    tmp.rename(dest)
    return dest


def onedrive_direct(url: str) -> str:
    return url if "download=1" in url else url + ("&" if "?" in url else "?") + "download=1"


def download(url: str, provider: str, dest: Path, progress: Callable[[str], None] = print) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if provider == "gdrive":
        try:
            import gdown
        except ImportError:
            raise SetupError("Google Drive downloads need gdown: pip install \"vivid2[setup]\" "
                             "(or download in a browser and use --plugin-zip)") from None
        out = gdown.download(url, str(dest), quiet=False, fuzzy=True, resume=True)
        if not out:
            raise SetupError("gdown could not download the file (quota or permissions?); use --plugin-zip instead")
        return dest
    if provider == "onedrive":
        url = onedrive_direct(url)
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as resp, open(dest, "ab" if have and resp.status == 206 else "wb") as f:
        total = int(resp.headers.get("Content-Length", 0)) + (have if resp.status == 206 else 0)
        done = have if resp.status == 206 else 0
        last = -1
        while True:
            block = resp.read(8 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            pct = int(100 * done / total) if total else -1
            if pct != last and pct % 5 == 0:
                progress(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({pct}%)")
                last = pct
    return dest


# -- orchestration ---------------------------------------------------------------

def setup(
    engine: str,
    plugin_dir: Optional[Path] = None,
    airsim_root: Optional[Path] = None,
    plugin_zip: Optional[Path] = None,
    do_download: bool = False,
    apply_patch: bool = False,
    force: bool = False,
    env_dir: Path = ENV_DIR,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    engines = load_engines()
    profile = engine_profile(engine, engines)
    chosen = [x for x in (plugin_dir, airsim_root, plugin_zip, do_download or None) if x]
    if len(chosen) != 1:
        raise SetupError("choose exactly one of --plugin-dir, --airsim-root, --plugin-zip, --download")

    source: Dict[str, Any]
    if plugin_dir:
        plugins = Path(plugin_dir).resolve()
        source = {"kind": "plugin-dir", "path": str(plugins)}
    elif airsim_root:
        root = Path(airsim_root).resolve()
        plugins = root / "unreal/Blocks/Plugins"
        if engine != "5.7" and not supports_ue58(root):
            patch = REPO / profile["patch"]
            if not apply_patch:
                raise SetupError(f"{root} does not support UE {engine} yet. Apply the patch:\n"
                                 f"  git -C \"{root}\" apply \"{patch}\"\nor rerun with --apply-patch")
            subprocess.run(["git", "-C", str(root), "apply", str(patch)], check=True)
            log(f"applied {patch.name} to {root}")
        if not has_team_weather_fix(root):
            patch = REPO / TEAM_WEATHER_PATCH
            if apply_patch:
                subprocess.run(["git", "-C", str(root), "apply", str(patch)], check=True)
                log(f"applied {patch.name} to {root}")
            else:  # single-drone runs don't need it
                log(f"warning: {root} lacks the team weather fix; team runs (fleet.size > 1) would see every "
                    f"drone's fog. Apply it: git -C \"{root}\" apply \"{patch}\" (or rerun with --apply-patch)")
        if not (plugins / "ProjectAirSim/SimLibs").exists():
            log(f"warning: no SimLibs in {plugins}/ProjectAirSim; run build.cmd in {root} first")
        source = {"kind": "airsim-root", "path": str(root)}
    else:
        z = profile.get("plugin_zip")
        if plugin_zip:
            zpath = Path(plugin_zip).resolve()
        else:
            if not z or not z.get("url"):
                raise SetupError(f"no download URL for UE {engine} in unreal/engines.json; use --plugin-zip")
            zpath = cache_root() / "downloads" / z["file_name"]
            log(f"downloading {z['file_name']} ({z.get('size', 0) / 1e9:.1f} GB) ...")
            download(z["url"], z.get("provider", "http"), zpath, log)
        log("verifying and extracting ...")
        # Check against the manifest hash unless the user supplied a differently named (own) zip.
        expected = (z or {}).get("sha256") or None
        if plugin_zip and z and zpath.name != z.get("file_name"):
            expected = None
        plugins = extract_plugin_zip(zpath, engine, expected)
        source = {"kind": "plugin-zip" if plugin_zip else "download", "path": str(zpath), "extracted": str(plugins)}

    link_plugins(plugins, env_dir, force)
    uproject = write_uproject(engine, env_dir, engines)

    eng_dir = find_engine_dir(engine)
    state = {
        "engine": engine,
        "engine_dir": str(eng_dir) if eng_dir else None,
        "engine_build_id": engine_build_id(eng_dir),
        "plugin_build_id": plugin_build_id(plugins),
        "simlibs_commit": simlibs_commit(plugins),
        "plugins_dir": str(plugins),
        "source": source,
        "uproject": str(uproject),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    (env_dir / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")

    if eng_dir is None:
        log(f"warning: UE {engine} not found (registry or Program Files); install it from the Epic launcher")
    elif state["plugin_build_id"] and state["plugin_build_id"] != state["engine_build_id"]:
        log(f"warning: plugin binaries were built for BuildId {state['plugin_build_id']}, engine is "
            f"{state['engine_build_id']}; UE will offer to rebuild the plugin (needs Visual Studio)")
    elif not state["plugin_build_id"]:
        log("note: plugin has no editor binaries yet; UE will compile it on first open (needs Visual Studio)")
    return state


def status(env_dir: Path = ENV_DIR) -> Optional[Dict[str, Any]]:
    p = env_dir / STATE_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
