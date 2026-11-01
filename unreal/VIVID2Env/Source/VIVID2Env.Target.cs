// VIVID 2 - MIT License.

using UnrealBuildTool;
using System.Collections.Generic;

public class VIVID2EnvTarget : TargetRules
{
	public VIVID2EnvTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.AddRange(new string[] { "VIVID2Env" });
	}
}
