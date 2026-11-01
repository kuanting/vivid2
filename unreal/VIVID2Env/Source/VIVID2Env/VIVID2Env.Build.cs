// VIVID 2 - MIT License.

using UnrealBuildTool;

public class VIVID2Env : ModuleRules
{
	public VIVID2Env(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		bEnableExceptions = true;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore" });
	}
}
