<#
.SYNOPSIS
  Package unreal/VIVID2Env as a standalone Windows build (no Unreal install needed to run it).

.DESCRIPTION
  Without -Group: cooks every map listed in Config/DefaultGame.ini (ForestSAR, ForestSAR_Open,
  ForestSAR_Dense, ForestFire, BlocksMap) plus the content spawned at runtime (hikers, decoys, fires,
  Blocks proxies, drone, weather), and archives the build to build\VIVID2Env-ue<engine>-Win64\Windows.
  With -Zip it also writes dist\VIVID2Env-ue<engine>-Win64-<commit>.zip to copy to other machines.

  With -Group <name>: cooks only that group's maps from unreal\VIVID2Env\package_groups.json (maps
  that share content ship together; every group also gets BlocksMap), plus the same runtime content,
  to build\VIVID2Env-<group>-ue<engine>-Win64 and dist\VIVID2Env-<group>-ue<engine>-Win64-<commit>.zip.
  Started without a map, any build opens BlocksMap.

  -NoDebugInfo leaves out the game's debug symbols (VIVID2Env.pdb, about 0.5 GB); crash dumps from
  such a build have no function names.

  The build contains third-party content (Fab packs) in cooked form only: never commit it,
  share it as a binary download.

  The editor may stay open: only the game target is compiled (-nocompileeditor), and the
  cook uses the editor binaries as they are. Close the editor if the cook runs out of memory.

  The pilot plugin for manual flight (vivid2 unreal setup --pilot-plugin) is not part of VIVID 2:
  packaging refuses while it is enabled, unless -AllowPrivatePlugins (a build for your own use).
  Run vivid2 unreal setup ... --no-pilot-plugin to package a build to share.

.EXAMPLE
  vivid2 unreal setup --engine 5.8 --download      # once: plugin + VIVID2Env.uproject for that engine
  powershell -ExecutionPolicy Bypass -File tools\package_env.ps1 -Engine 5.8 -Zip
  # then start a map:
  build\VIVID2Env-ue5.8-Win64\Windows\VIVID2Env.exe /Game/VIVID2/Maps/ForestSAR_Open -windowed -ResX=1280 -ResY=720

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\package_env.ps1 -Group forest -NoDebugInfo -Zip
  # -> build\VIVID2Env-forest-ue5.8-Win64, dist\VIVID2Env-forest-ue5.8-Win64-<commit>.zip
#>
param(
    [ValidateSet("5.7", "5.8")]
    [string]$Engine = "5.8",
    [string]$UERoot = "",
    [ValidateSet("Development", "Shipping", "DebugGame")]
    [string]$Config = "Development",
    [string]$OutDir = "",
    [string]$Group = "",
    [switch]$NoDebugInfo,
    [switch]$Zip,
    [switch]$AllowPrivatePlugins
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $UERoot) { $UERoot = "C:\Program Files\Epic Games\UE_$Engine" }

# The maps to cook: a group's (package_groups.json), or MapsToCook from DefaultGame.ini (no -Group).
$mapRoot = "/Game/VIVID2/Maps/"
if ($Group) {
    $groups = Get-Content (Join-Path $repo "unreal\VIVID2Env\package_groups.json") -Raw | ConvertFrom-Json
    $names = @($groups.groups.PSObject.Properties.Name)
    if ($names -notcontains $Group) { throw "unknown group '$Group'; package_groups.json has: $($names -join ', ')" }
    $maps = @($groups.groups.$Group) + @($groups.every_build | Where-Object { @($groups.groups.$Group) -notcontains $_ })
    $buildName = "VIVID2Env-$Group-ue$Engine-Win64"
} else {
    $ini = Get-Content (Join-Path $repo "unreal\VIVID2Env\Config\DefaultGame.ini")
    $maps = @($ini | Select-String '^\+MapsToCook=\(FilePath="([^"]+)"\)' | ForEach-Object { $_.Matches[0].Groups[1].Value.Replace($mapRoot, "") })
    $buildName = "VIVID2Env-ue$Engine-Win64"
}
if (-not $OutDir) { $OutDir = Join-Path $repo "build\$buildName" }
$uproject = Join-Path $repo "unreal\VIVID2Env\VIVID2Env.uproject"
if (-not (Test-Path $uproject)) { throw "VIVID2Env.uproject not found; run: vivid2 unreal setup --engine $Engine ..." }
$assoc = (Get-Content $uproject -Raw | ConvertFrom-Json).EngineAssociation
if ($assoc -ne $Engine) { throw "VIVID2Env.uproject is set up for UE $assoc; rerun: vivid2 unreal setup --engine $Engine ..." }
$pilot = (Get-Content $uproject -Raw | ConvertFrom-Json).Plugins | Where-Object { $_.Name -eq "FpvPilot" -and $_.Enabled }
if ($pilot -and -not $AllowPrivatePlugins) {
    throw "The pilot plugin (FpvPilot) is enabled; it is not part of VIVID 2. Unlink it first (vivid2 unreal setup --engine $Engine ... --no-pilot-plugin) or pass -AllowPrivatePlugins for a build you keep."
}

$uat = Join-Path $UERoot "Engine\Build\BatchFiles\RunUAT.bat"
if (-not (Test-Path $uat)) { throw "RunUAT not found at $uat (set -UERoot)" }
$project = (Resolve-Path $uproject).Path

# Quote "-name=$value" arguments: PowerShell passes an unquoted -clientconfig=$Config literally.
$extra = @()
# -MapsToCook overrides the ini's MapsToCook; DirectoriesToAlwaysCook and the default map
# (BlocksMap, in every group) are still cooked.
if ($Group) { $extra += "-MapsToCook=" + (($maps | ForEach-Object { "$mapRoot$_" }) -join "+") }
if ($NoDebugInfo) { $extra += "-nodebuginfo" }
& $uat BuildCookRun "-project=$project" -platform=Win64 "-clientconfig=$Config" `
    -build -nocompileeditor -cook -stage -pak -prereqs -archive "-archivedirectory=$OutDir" `
    -unattended -noP4 -utf8output @extra
if ($LASTEXITCODE -ne 0) { throw "BuildCookRun failed ($LASTEXITCODE)" }

# Record what was packaged, next to the exe: VIVID 2 commit, engine, config, maps, date.
$build = Join-Path $OutDir "Windows"
$commit = (git -C $repo rev-parse --short HEAD).Trim()
$modified = @(git -C $repo status --porcelain -- unreal/VIVID2Env/Content/VIVID2 unreal/VIVID2Env/Config unreal/VIVID2Env/Source unreal/VIVID2Env/package_groups.json)
$record = [ordered]@{ vivid2_commit = $commit; uncommitted_changes = $modified; engine = $Engine; config = $Config
   group = $(if ($Group) { $Group } else { "all" }); maps = $maps; debug_info = -not $NoDebugInfo
   packaged = (Get-Date -Format "s") }
$record | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $build "vivid2_build.json")
Write-Host "Packaged UE $Engine build ($($maps -join ', ')) to $build"

if ($Zip) {
    $dist = Join-Path $repo "dist"
    New-Item -ItemType Directory -Force $dist | Out-Null
    $zipPath = Join-Path $dist "$buildName-$commit.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath }
    # tar.exe (bsdtar, in Windows 10+) because Compress-Archive fails on files over 2 GB (the .pak/.ucas)
    tar.exe -a -c -f $zipPath -C $build .
    if ($LASTEXITCODE -ne 0) { throw "zip failed ($LASTEXITCODE)" }
    $hash = (Get-FileHash $zipPath -Algorithm SHA256).Hash.ToLower()
    Set-Content -Encoding ascii "$zipPath.sha256" "$hash  $(Split-Path $zipPath -Leaf)"
    Write-Host "Zipped to $zipPath ($([math]::Round((Get-Item $zipPath).Length / 1GB, 2)) GB, sha256 $hash)"
}
