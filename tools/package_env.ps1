<#
.SYNOPSIS
  Package unreal/VIVID2Env as a standalone (headless-capable) Windows build.

.EXAMPLE
  vivid2 unreal setup --engine 5.8 --download      # once: plugin + VIVID2Env.uproject for that engine
  powershell -ExecutionPolicy Bypass -File tools\package_env.ps1 -Engine 5.8 -Config Development
  # then run:  build\VIVID2Env-ue5.8-Win64\Windows\VIVID2Env.exe -RenderOffScreen   (see ProjectAirSim docs/command_line_switches.md)
#>
param(
    [ValidateSet("5.7", "5.8")]
    [string]$Engine = "5.8",
    [string]$UERoot = "",
    [ValidateSet("Development", "Shipping", "DebugGame")]
    [string]$Config = "Development",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
if (-not $UERoot) { $UERoot = "C:\Program Files\Epic Games\UE_$Engine" }
if (-not $OutDir) { $OutDir = Join-Path $PSScriptRoot "..\build\VIVID2Env-ue$Engine-Win64" }
$uproject = Join-Path $PSScriptRoot "..\unreal\VIVID2Env\VIVID2Env.uproject"
if (-not (Test-Path $uproject)) { throw "VIVID2Env.uproject not found; run: vivid2 unreal setup --engine $Engine ..." }
$assoc = (Get-Content $uproject -Raw | ConvertFrom-Json).EngineAssociation
if ($assoc -ne $Engine) { throw "VIVID2Env.uproject is set up for UE $assoc; rerun: vivid2 unreal setup --engine $Engine ..." }

$uat = Join-Path $UERoot "Engine\Build\BatchFiles\RunUAT.bat"
if (-not (Test-Path $uat)) { throw "RunUAT not found at $uat (set -UERoot)" }
$project = (Resolve-Path $uproject).Path

& $uat BuildCookRun -project="$project" -platform=Win64 -clientconfig=$Config `
    -build -cook -stage -pak -archive -archivedirectory="$OutDir" -noP4 -utf8output
if ($LASTEXITCODE -ne 0) { throw "BuildCookRun failed ($LASTEXITCODE)" }
Write-Host "Packaged UE $Engine build to $OutDir"
