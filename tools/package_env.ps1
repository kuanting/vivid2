<#
.SYNOPSIS
  Package unreal/VIVID2Env as a standalone Windows build (no Unreal install needed to run it).

.DESCRIPTION
  Cooks the maps listed in Config/DefaultGame.ini (ForestSAR, ForestSAR_Open, ForestSAR_Dense,
  ForestFire) plus the content spawned at runtime (hikers, decoys, fires, drone, weather), and archives the
  build to build\VIVID2Env-ue<engine>-Win64\Windows. With -Zip it also writes
  dist\VIVID2Env-ue<engine>-Win64-<commit>.zip to copy to other machines.

  The build contains third-party content (Fab packs) in cooked form only: never commit it,
  share it as a binary download.

  The editor may stay open: only the game target is compiled (-nocompileeditor), and the
  cook uses the editor binaries as they are. Close the editor if the cook runs out of memory.

.EXAMPLE
  vivid2 unreal setup --engine 5.8 --download      # once: plugin + VIVID2Env.uproject for that engine
  powershell -ExecutionPolicy Bypass -File tools\package_env.ps1 -Engine 5.8 -Zip
  # then start a map (default ForestSAR):
  build\VIVID2Env-ue5.8-Win64\Windows\VIVID2Env.exe /Game/VIVID2/Maps/ForestSAR_Open -windowed -ResX=1280 -ResY=720
#>
param(
    [ValidateSet("5.7", "5.8")]
    [string]$Engine = "5.8",
    [string]$UERoot = "",
    [ValidateSet("Development", "Shipping", "DebugGame")]
    [string]$Config = "Development",
    [string]$OutDir = "",
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $UERoot) { $UERoot = "C:\Program Files\Epic Games\UE_$Engine" }
if (-not $OutDir) { $OutDir = Join-Path $repo "build\VIVID2Env-ue$Engine-Win64" }
$uproject = Join-Path $repo "unreal\VIVID2Env\VIVID2Env.uproject"
if (-not (Test-Path $uproject)) { throw "VIVID2Env.uproject not found; run: vivid2 unreal setup --engine $Engine ..." }
$assoc = (Get-Content $uproject -Raw | ConvertFrom-Json).EngineAssociation
if ($assoc -ne $Engine) { throw "VIVID2Env.uproject is set up for UE $assoc; rerun: vivid2 unreal setup --engine $Engine ..." }

$uat = Join-Path $UERoot "Engine\Build\BatchFiles\RunUAT.bat"
if (-not (Test-Path $uat)) { throw "RunUAT not found at $uat (set -UERoot)" }
$project = (Resolve-Path $uproject).Path

# Quote "-name=$value" arguments: PowerShell passes an unquoted -clientconfig=$Config literally.
& $uat BuildCookRun "-project=$project" -platform=Win64 "-clientconfig=$Config" `
    -build -nocompileeditor -cook -stage -pak -prereqs -archive "-archivedirectory=$OutDir" `
    -unattended -noP4 -utf8output
if ($LASTEXITCODE -ne 0) { throw "BuildCookRun failed ($LASTEXITCODE)" }

# Record what was packaged, next to the exe: VIVID 2 commit, engine, config, date.
$build = Join-Path $OutDir "Windows"
$commit = (git -C $repo rev-parse --short HEAD).Trim()
$modified = @(git -C $repo status --porcelain -- unreal/VIVID2Env/Content/VIVID2 unreal/VIVID2Env/Config unreal/VIVID2Env/Source)
@{ vivid2_commit = $commit; uncommitted_changes = $modified; engine = $Engine; config = $Config
   packaged = (Get-Date -Format "s") } | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $build "vivid2_build.json")
Write-Host "Packaged UE $Engine build to $build"

if ($Zip) {
    $dist = Join-Path $repo "dist"
    New-Item -ItemType Directory -Force $dist | Out-Null
    $zipPath = Join-Path $dist "VIVID2Env-ue$Engine-Win64-$commit.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath }
    # tar.exe (bsdtar, in Windows 10+) because Compress-Archive fails on files over 2 GB (the .pak/.ucas)
    tar.exe -a -c -f $zipPath -C $build .
    if ($LASTEXITCODE -ne 0) { throw "zip failed ($LASTEXITCODE)" }
    $hash = (Get-FileHash $zipPath -Algorithm SHA256).Hash.ToLower()
    Set-Content -Encoding ascii "$zipPath.sha256" "$hash  $(Split-Path $zipPath -Leaf)"
    Write-Host "Zipped to $zipPath ($([math]::Round((Get-Item $zipPath).Length / 1GB, 2)) GB, sha256 $hash)"
}
