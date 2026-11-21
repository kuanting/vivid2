"""Packaged builds: the build groups, the cook list and the default map agree."""
import json
import re

from vivid2.scenario import Scenario

MAPS = "/Game/VIVID2/Maps/"


def _ini_values(path, key):
    return re.findall(rf'^\+?{key}=(.*)$', path.read_text(encoding="utf-8"), re.M)


def _maps_to_cook(root):
    lines = _ini_values(root / "unreal/VIVID2Env/Config/DefaultGame.ini", "MapsToCook")
    return [re.fullmatch(r'\(FilePath="([^"]+)"\)', v).group(1) for v in lines]


def test_the_build_groups_cover_the_cook_list_once(root):
    doc = json.loads((root / "unreal/VIVID2Env/package_groups.json").read_text(encoding="utf-8"))
    cooked = _maps_to_cook(root)
    assert all(m.startswith(MAPS) for m in cooked)
    in_groups = [m for maps in doc["groups"].values() for m in maps]
    assert len(in_groups) == len(set(in_groups)), "a map is in two groups"
    assert not set(in_groups) & set(doc["every_build"])
    assert sorted(in_groups + doc["every_build"]) == sorted(m[len(MAPS):] for m in cooked)


def test_every_build_has_the_default_map(root):
    doc = json.loads((root / "unreal/VIVID2Env/package_groups.json").read_text(encoding="utf-8"))
    (default,) = _ini_values(root / "unreal/VIVID2Env/Config/DefaultEngine.ini", "GameDefaultMap")
    assert default.startswith(MAPS) and default[len(MAPS):] in doc["every_build"]


def test_every_scenario_map_is_cooked(root):
    cooked = {m[len(MAPS):] for m in _maps_to_cook(root)}
    for p in (root / "scenarios").rglob("*.jsonc"):
        m = Scenario.load(p).spec["scene"].get("map")
        # the legacy blocks_* scenarios run in Project AirSim's own Blocks project, not in VIVID2Env
        assert m is None or m in cooked or m == "BlocksMap", f"{p.name}: map {m!r} is in no packaged build"
