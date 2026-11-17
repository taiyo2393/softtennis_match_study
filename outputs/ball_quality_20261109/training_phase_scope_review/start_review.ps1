$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\third_party\racketvision-env\Scripts\python.exe'
$reviewScript = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\ball_quality_20261103\serve_training_phase_scope.py'))
$scriptHash = (Get-FileHash -LiteralPath $reviewScript -Algorithm SHA256).Hash.ToLowerInvariant()
$statePath = Join-Path $PSScriptRoot 'server_state.json'
if (Test-Path -LiteralPath $statePath) {
    $priorState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    try {
        $data = Invoke-RestMethod -Uri ($priorState.url + 'api/data') -TimeoutSec 2
        if ($data.samples.Count -eq 123 -and $priorState.server_script_sha256 -eq $scriptHash) { Write-Output $priorState.url; return }
    } catch { }
}
$launchTimeUtc = [DateTime]::UtcNow
$process = Start-Process -FilePath $pythonExe -ArgumentList @($reviewScript,'--port','0') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'server_stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'server_stderr.log')
$deadline = (Get-Date).AddSeconds(25)
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) { throw (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'server_stderr.log') -Raw) }
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ((Get-Item -LiteralPath $statePath).LastWriteTimeUtc -ge $launchTimeUtc -and $state.server_script_sha256 -eq $scriptHash) { Write-Output $state.url; return }
    }
    Start-Sleep -Milliseconds 150
}
throw 'Server startup timed out; check server_stderr.log.'
