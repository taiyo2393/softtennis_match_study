$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\third_party\racketvision-env\Scripts\python.exe'
$reviewScript = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\ball_quality_20261103\serve_reserved_ball_sequence_review_fit.py'))
$expectedScriptHash = (Get-FileHash -LiteralPath $reviewScript -Algorithm SHA256).Hash.ToLowerInvariant()
$statePath = Join-Path $PSScriptRoot 'server_state.json'
if (Test-Path -LiteralPath $statePath) {
    $savedState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    try {
        $response = Invoke-RestMethod -Uri ($savedState.url + 'api/cases') -TimeoutSec 2
        if ($response.cases.Count -eq 52 -and $savedState.server_script_sha256 -eq $expectedScriptHash) {
            Write-Output $savedState.url
            return
        }
    } catch { }
}
$launchBeginUtc = [DateTime]::UtcNow
$expectedScriptHash = (Get-FileHash -LiteralPath $reviewScript -Algorithm SHA256).Hash.ToLowerInvariant()
$reviewProcess = Start-Process -FilePath $pythonExe -ArgumentList @($reviewScript, '--port', '0') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'server_stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'server_stderr.log')
$deadline = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $deadline) {
    if ($reviewProcess.HasExited) {
        throw (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'server_stderr.log') -Raw)
    }
    if (Test-Path -LiteralPath $statePath) {
        $startedState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        # A virtual-environment launcher may use a child Python PID. Verify the
        # newly written readiness record and script hash instead of PID equality.
        if ((Get-Item -LiteralPath $statePath).LastWriteTimeUtc -ge $launchBeginUtc -and $startedState.server_script_sha256 -eq $expectedScriptHash) {
            Write-Output $startedState.url
            return
        }
    }
    Start-Sleep -Milliseconds 150
}
throw 'Review server did not report readiness within20seconds. Check server_stderr.log.'
