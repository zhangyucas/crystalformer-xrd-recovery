param(
    [string]$Distro = "Ubuntu",
    [string]$OutputDir,
    [double]$MinFreePhysicalGB = 1.5,
    [double]$MinFreeCommitGB = 3.0,
    [int]$IntervalSeconds = 5,
    [int]$ConsecutiveLowChecks = 2,
    [int]$StartupGraceSeconds = 120,
    [string]$WatchPattern = "crystalformer.cli.train_ppo",
    [string]$KillPattern = "crystalformer.cli.train_ppo",
    [int]$MaxHours = 8
)

$ErrorActionPreference = "Stop"
$output = [System.IO.Path]::GetFullPath($OutputDir)
[System.IO.Directory]::CreateDirectory($output) | Out-Null
$telemetry = Join-Path $output "windows_host_memory.csv"
$stopReason = Join-Path $output "windows_host_stop_reason.json"
if (Test-Path -LiteralPath $stopReason) {
    Remove-Item -LiteralPath $stopReason -Force
}
"unix_time,free_physical_gib,free_commit_gib,total_commit_gib" | Set-Content -Path $telemetry -Encoding ASCII

$deadline = (Get-Date).AddHours($MaxHours)
$startupDeadline = (Get-Date).AddSeconds($StartupGraceSeconds)
$lowChecks = 0
$sawTraining = $false
while ((Get-Date) -lt $deadline) {
    $os = Get-CimInstance Win32_OperatingSystem
    $freePhysical = [double]$os.FreePhysicalMemory / 1MB
    $freeCommit = [double]$os.FreeVirtualMemory / 1MB
    $totalCommit = [double]$os.TotalVirtualMemorySize / 1MB
    $unixTime = [DateTimeOffset]::Now.ToUnixTimeMilliseconds() / 1000.0
    ("{0:F3},{1:F3},{2:F3},{3:F3}" -f $unixTime,$freePhysical,$freeCommit,$totalCommit) |
        Add-Content -Path $telemetry -Encoding ASCII

    if ($freePhysical -lt $MinFreePhysicalGB -or $freeCommit -lt $MinFreeCommitGB) {
        $lowChecks += 1
    } else {
        $lowChecks = 0
    }
    if ($lowChecks -ge $ConsecutiveLowChecks) {
        $reason = [ordered]@{
            reason = "windows_host_memory_stop"
            free_physical_gib = $freePhysical
            free_commit_gib = $freeCommit
            min_free_physical_gib = $MinFreePhysicalGB
            min_free_commit_gib = $MinFreeCommitGB
            consecutive_low_checks = $lowChecks
            unix_time = $unixTime
        }
        $reason | ConvertTo-Json | Set-Content -Path $stopReason -Encoding ASCII
        & wsl.exe -d $Distro -- pkill -TERM -f $KillPattern
        exit 75
    }

    $running = & wsl.exe -d $Distro -- pgrep -f $WatchPattern
    if ($LASTEXITCODE -eq 0) {
        $sawTraining = $true
    } elseif ($sawTraining -or (Get-Date) -ge $startupDeadline) {
        exit 0
    }
    Start-Sleep -Seconds $IntervalSeconds
}
exit 0
