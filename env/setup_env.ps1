<#
.SYNOPSIS
    Create (or update) and activate the SBP Studio Conda environment on Windows.

.DESCRIPTION
    Resolves environment.yml next to this script, creates the env (or updates it
    in place with --prune if it already exists), then activates it.

    `conda activate` only affects the shell it runs in. To stay in the env after
    the script ends, DOT-SOURCE it:

        . .\env\setup_env.ps1          # keeps your shell in the env
        .\env\setup_env.ps1            # runs, then prints the activate command

.EXAMPLE
    . .\env\setup_env.ps1
#>
$ErrorActionPreference = "Stop"

# Resolve this script's folder so environment.yml is found from any CWD.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile   = Join-Path $ScriptDir "environment.yml"

# Read the env name from the YAML (fallback: sbp_studio).
$EnvName = "sbp_studio"
if (Test-Path $EnvFile) {
    $nameLine = Select-String -Path $EnvFile -Pattern '^\s*name:\s*(\S+)' | Select-Object -First 1
    if ($nameLine) { $EnvName = $nameLine.Matches[0].Groups[1].Value }
}

Write-Host ">> SBP Studio environment setup"
Write-Host "   env file : $EnvFile"
Write-Host "   env name : $EnvName"

# 1) conda must be available.
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "'conda' was not found on PATH. Install Miniconda/Anaconda and open an Anaconda PowerShell Prompt first: https://docs.conda.io/en/latest/miniconda.html"
    return
}
if (-not (Test-Path $EnvFile)) {
    Write-Error "environment.yml not found at $EnvFile"
    return
}

# 2) Create the env, or update it in place if it already exists. `conda env list`
#    is matched on the first whitespace-delimited token (the env name).
$existing = (conda env list) | ForEach-Object { ($_ -split '\s+')[0] }
if ($existing -contains $EnvName) {
    Write-Host ">> Environment '$EnvName' exists - updating from environment.yml ..."
    conda env update -n $EnvName -f $EnvFile --prune
} else {
    Write-Host ">> Creating environment '$EnvName' from environment.yml ..."
    conda env create -f $EnvFile
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "conda env create/update failed (exit $LASTEXITCODE)."
    return
}

# 3) Activate (works when conda is initialised for PowerShell).
Write-Host ">> Activating '$EnvName' ..."
conda activate $EnvName

Write-Host ""
Write-Host ">> Done. Verify hardware acceleration with:"
Write-Host "     python -m sbp_studio.cli.main accel"
Write-Host ">> Launch the GUI with:"
Write-Host "     python -m sbp_studio.gui"
Write-Host ""
Write-Host "NOTE: if your prompt does not show ($EnvName), dot-source this script"
Write-Host "      so activation sticks:   . .\env\setup_env.ps1"
Write-Host "      or run manually:        conda activate $EnvName"
