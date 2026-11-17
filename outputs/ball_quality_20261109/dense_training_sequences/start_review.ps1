$taskFolder = Split-Path -Parent $MyInvocation.MyCommand.Path
$taskStatePath = Join-Path $taskFolder 'server_state.json'
$taskRunning = $false
if (Test-Path -LiteralPath $taskStatePath) {
  $taskState = Get-Content -LiteralPath $taskStatePath -Raw | ConvertFrom-Json
  try { $taskResponse = Invoke-WebRequest -Uri ($taskState.url + 'api/cases') -TimeoutSec 3; $taskRunning = ($taskResponse.StatusCode -eq 200) } catch {}
}
if (-not $taskRunning) {
  Start-Process -FilePath 'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\third_party\racketvision-env\Scripts\python.exe' -ArgumentList @('-X','utf8', ('"' + (Join-Path $taskFolder 'serve_review.py') + '"')) -WorkingDirectory $taskFolder -WindowStyle Hidden
  for ($taskAttempt=0; $taskAttempt -lt 60; $taskAttempt++) {
    Start-Sleep -Milliseconds 500
    if (Test-Path -LiteralPath $taskStatePath) {
      $taskState = Get-Content -LiteralPath $taskStatePath -Raw | ConvertFrom-Json
      try { $taskResponse = Invoke-WebRequest -Uri ($taskState.url + 'api/cases') -TimeoutSec 2; if ($taskResponse.StatusCode -eq 200) { $taskRunning = $true; break } } catch {}
    }
  }
}
if (-not $taskRunning) { throw 'Review server did not become ready' }
Start-Process $taskState.url
Write-Output $taskState.url
