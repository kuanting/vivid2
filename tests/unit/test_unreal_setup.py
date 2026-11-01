import json
import zipfile

import pytest

from vivid2 import unreal_setup as us


def _fake_plugins(root, build_id="55116800"):
    """Minimal plugin tree shaped like a built Project AirSim plugin."""
    pas = root / "ProjectAirSim"
    (pas / "Binaries/Win64").mkdir(parents=True)
    (pas / "Binaries/Win64/UnrealEditor.modules").write_text(json.dumps({"BuildId": build_id, "Modules": {}}))
    (pas / "Binaries/Win64/UnrealEditor-ProjectAirSim.dll").write_bytes(b"dll")
    (pas / "Binaries/Win64/UnrealEditor-ProjectAirSim.pdb").write_bytes(b"pdb")
    (pas / "Intermediate/Build").mkdir(parents=True)
    (pas / "Intermediate/Build/junk.obj").write_bytes(b"x" * 100)
    (pas / "SimLibs/eigen/.git").mkdir(parents=True)
    (pas / "SimLibs/eigen/.git/index").write_bytes(b"git")
    (pas / "SimLibs/eigen/Eigen.h").write_text("// eigen")
    (pas / "SimLibs/commit_hash.txt").write_text("abc123\n")
    (pas / "ProjectAirSim.uplugin").write_text("{}")
    (root / "Drone/Content").mkdir(parents=True)
    (root / "Drone/Drone.uplugin").write_text("{}")
    return root


def _env_dir(tmp_path, root):
    env = tmp_path / "VIVID2Env"
    env.mkdir()
    (env / "VIVID2Env.uproject.template.json").write_text(
        (root / "unreal/VIVID2Env/VIVID2Env.uproject.template.json").read_text(encoding="utf-8"))
    return env


def test_engines_manifest_profiles():
    engines = us.load_engines()
    assert {"5.7", "5.8"} <= set(engines["engines"])
    assert engines["engines"]["5.7"]["extra_plugins"] == []
    assert "ModelContextProtocol" in engines["engines"]["5.8"]["extra_plugins"]
    assert (us.REPO / engines["engines"]["5.8"]["patch"]).exists()
    with pytest.raises(us.SetupError):
        us.engine_profile("5.6", engines)


@pytest.mark.parametrize("engine", ["5.7", "5.8"])
def test_uproject_rendered_per_engine(tmp_path, root, engine):
    env = _env_dir(tmp_path, root)
    up = json.loads(us.write_uproject(engine, env).read_text())
    names = [p["Name"] for p in up["Plugins"]]
    assert up["EngineAssociation"] == engine
    assert "ProjectAirSim" in names
    assert ("ModelContextProtocol" in names) == (engine == "5.8")
    assert len(names) == len(set(names))


def test_build_id_and_commit(tmp_path):
    plugins = _fake_plugins(tmp_path / "Plugins")
    assert us.plugin_build_id(plugins) == "55116800"
    assert us.simlibs_commit(plugins) == "abc123"
    assert us.read_build_id(tmp_path / "missing.modules") is None


def test_supports_ue58(tmp_path):
    hdr = tmp_path / "unreal/Blocks/Plugins/ProjectAirSim/Source/ProjectAirSim/Private/UnrealCompatibility.h"
    hdr.parent.mkdir(parents=True)
    hdr.write_text("#define UE_IS_5_7 (1)\n")
    assert not us.supports_ue58(tmp_path)
    hdr.write_text("#define UE_IS_5_8 (1)\n")
    assert us.supports_ue58(tmp_path)


def test_pack_excludes_intermediate_git_and_pdb(tmp_path):
    plugins = _fake_plugins(tmp_path / "Plugins")
    info = us.pack_plugin(plugins, "5.8", tmp_path / "dist", strip_pdb=True)
    names = zipfile.ZipFile(info["zip"]).namelist()
    assert "ProjectAirSim/ProjectAirSim.uplugin" in names and "Drone/Drone.uplugin" in names
    assert "ProjectAirSim/SimLibs/eigen/Eigen.h" in names
    assert not any("Intermediate" in n or "/.git/" in n or n.endswith(".pdb") for n in names)
    assert "VIVID2_PLUGIN.json" in names
    assert info["sha256"] == us.sha256_file(tmp_path / "dist" / info["file_name"])
    assert (tmp_path / "dist" / (info["file_name"] + ".sha256")).read_text().startswith(info["sha256"])


def test_zip_roundtrip_and_sha_check(tmp_path):
    plugins = _fake_plugins(tmp_path / "Plugins")
    info = us.pack_plugin(plugins, "5.8", tmp_path / "dist")
    cache = tmp_path / "cache"
    with pytest.raises(us.SetupError, match="SHA-256 mismatch"):
        us.extract_plugin_zip(tmp_path / "dist" / info["file_name"], "5.8", "0" * 64, cache)
    out = us.extract_plugin_zip(tmp_path / "dist" / info["file_name"], "5.8", info["sha256"], cache)
    assert (out / "ProjectAirSim/ProjectAirSim.uplugin").exists()
    assert us.plugin_build_id(out) == "55116800"
    # second call reuses the cache
    assert us.extract_plugin_zip(tmp_path / "dist" / info["file_name"], "5.8", None, cache) == out


def test_zip_with_path_traversal_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(us.SetupError, match="unsafe"):
        us.extract_plugin_zip(bad, "5.8", None, tmp_path / "cache")


def test_setup_from_plugin_dir_links_and_records(tmp_path, root):
    plugins = _fake_plugins(tmp_path / "Plugins")
    env = _env_dir(tmp_path, root)
    state = us.setup("5.8", plugin_dir=plugins, env_dir=env, log=lambda s: None)
    assert (env / "Plugins/ProjectAirSim/ProjectAirSim.uplugin").exists()
    assert us._is_link(env / "Plugins/ProjectAirSim")
    assert json.loads((env / "VIVID2Env.uproject").read_text())["EngineAssociation"] == "5.8"
    assert us.status(env)["plugin_build_id"] == "55116800" == state["plugin_build_id"]
    # re-running (e.g. switching engines) replaces the links
    us.setup("5.7", plugin_dir=plugins, env_dir=env, log=lambda s: None)
    assert json.loads((env / "VIVID2Env.uproject").read_text())["EngineAssociation"] == "5.7"


def test_setup_requires_exactly_one_source(tmp_path, root):
    with pytest.raises(us.SetupError, match="exactly one"):
        us.setup("5.8", env_dir=_env_dir(tmp_path, root), log=lambda s: None)


def test_onedrive_direct_link():
    assert us.onedrive_direct("https://1drv.ms/u/s!abc") == "https://1drv.ms/u/s!abc?download=1"
    assert us.onedrive_direct("https://x/y?e=1") == "https://x/y?e=1&download=1"
