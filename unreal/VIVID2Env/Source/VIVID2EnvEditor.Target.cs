// VIVID 2 - MIT License.

using UnrealBuildTool;
using System.Collections.Generic;

public class VIVID2EnvEditorTarget : TargetRules
{
	public VIVID2EnvEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.AddRange(new string[] { "VIVID2Env" });
	}
}
