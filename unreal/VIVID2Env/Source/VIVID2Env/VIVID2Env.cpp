// VIVID 2 - MIT License.
//
// Map/content-only game module. It exists so the ProjectAirSim plugin can be
// compiled when packaging a standalone build (see ProjectAirSim docs/use_plugin.md).

#include "VIVID2Env.h"

#include "Modules/ModuleManager.h"

IMPLEMENT_PRIMARY_GAME_MODULE(FDefaultGameModuleImpl, VIVID2Env, "VIVID2Env");
