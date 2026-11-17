$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\third_party\racketvision-env\Scripts\python.exe'
$reviewScript = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\ball_quality_20261103\serve_temporal_guard_packet.py'))
$scriptHash = (Get-FileHash -LiteralPath $reviewScript -Algorithm SHA256).Hash.ToLowerInvariant()
$statePath = Join-Path $PSScriptRoot 'server_state.json'
if (Test-Path -LiteralPath $statePath) {
    $saved = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    try {
        $response = Invoke-RestMethod -Uri ($saved.url + 'api/cases') -TimeoutSec 2
        if ($response.cases.Count -eq 26 -and $saved.server_script_sha256 -eq $scriptHash) {
            Write-Output $saved.url
            return
        }
    } catch { }
}
$launchedUtc = [DateTime]::UtcNow
$reviewProcess = Start-Process -FilePath $pythonExe -ArgumentList @($reviewScript, '--port', '0') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'server_stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'server_stderr.log')
$deadline = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $deadline) {
    if ($reviewProcess.HasExited) { throw (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'server_stderr.log') -Raw) }
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ((Get-Item -LiteralPath $statePath).LastWriteTimeUtc -ge $launchedUtc -and $state.server_script_sha256 -eq $scriptHash) {
            Write-Output $state.url
            return
        }
    }
    Start-Sleep -Milliseconds 150
}
throw '確認画面の起動に失敗しました。server_stderr.logを確認してください。'
