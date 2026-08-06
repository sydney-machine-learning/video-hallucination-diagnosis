param(
    [string]$JobId = "8919110",
    [string]$FullJobId = "8919110.kman.restech.unsw.edu.au",
    [int]$IntervalSeconds = 180,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$RouteRoot = Split-Path -Parent $PSScriptRoot
$ResultRoot = Join-Path $RouteRoot "results\perception_confirmation"
$LogPath = Join-Path $RouteRoot "results\perception_katana_watch.log"
$StatusPath = Join-Path $RouteRoot "results\PERCEPTION_KATANA_WATCH_STATUS.json"
$PidPath = Join-Path $RouteRoot "results\perception_katana_watch.pid"
$KeyPath = Join-Path $env:USERPROFILE ".ssh\codex_katana_ed25519"
$RemoteHost = "129.94.146.61"
$RemoteUser = "z5611125"
$RemoteTarget = "${RemoteUser}@${RemoteHost}"
$RemoteRunRoot = "/srv/scratch/z5611125/sicr/trace_rc_perception_valid500"
$RemoteAnalysis = "/home/z5611125/trace_rc_route/results_fresh/perception_test_confirmation/${FullJobId}"

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent), $ResultRoot | Out-Null

function Write-WatchLog {
    param([string]$Message)
    $line = "{0}`t{1}" -f (Get-Date -Format o), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding utf8
    Write-Output $line
}

function Write-WatchStatus {
    param(
        [string]$State,
        [string]$Comment = "",
        [string]$EstimatedStart = "",
        [string]$Detail = ""
    )
    [ordered]@{
        schema_version = "trace-rc-perception-katana-watch-1.0"
        updated_at = (Get-Date -Format o)
        job_id = $FullJobId
        state = $State
        comment = $Comment
        estimated_start = $EstimatedStart
        detail = $Detail
        log_path = $LogPath
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

function Invoke-KatanaSsh {
    param([string]$RemoteCommand)
    $savedPreference = $ErrorActionPreference
    $output = @()
    $exitCode = 255
    try {
        # A transient SSH stderr record must be handled as a retry, not as a
        # terminating PowerShell error that kills the long-running watcher.
        $ErrorActionPreference = "Continue"
        $output = & ssh.exe -i $KeyPath `
            -o HostKeyAlias=katana.restech.unsw.edu.au `
            -o BatchMode=yes -o ConnectTimeout=20 `
            $RemoteTarget $RemoteCommand 2>&1
        $exitCode = $LASTEXITCODE
    }
    catch {
        $output = @($_.Exception.Message)
        $exitCode = 255
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }
    return [pscustomobject]@{ ExitCode = $exitCode; Output = @($output) }
}

function Get-Field {
    param([string[]]$Lines, [string]$Name)
    $match = $Lines | Select-String -Pattern ("^\s*" + [regex]::Escape($Name) + "\s*=\s*(.+)$") | Select-Object -First 1
    if ($match) { return $match.Matches[0].Groups[1].Value.Trim() }
    return ""
}

if (-not (Test-Path -LiteralPath $KeyPath)) {
    throw "Katana SSH key is missing: $KeyPath"
}

if (-not $Once -and (Test-Path -LiteralPath $PidPath)) {
    $oldPid = 0
    [void][int]::TryParse((Get-Content -LiteralPath $PidPath -Raw).Trim(), [ref]$oldPid)
    if ($oldPid -gt 0 -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        Write-WatchLog "WATCH_ALREADY_RUNNING pid=$oldPid"
        exit 17
    }
}

if (-not $Once) {
    Set-Content -LiteralPath $PidPath -Value $PID -Encoding ascii
}

try {
    Write-WatchLog "WATCH_START job=$FullJobId interval_seconds=$IntervalSeconds once=$Once"
    while ($true) {
        $query = Invoke-KatanaSsh "qstat -fx $JobId 2>&1"
        if ($query.ExitCode -ne 0) {
            Write-WatchStatus -State "connection_retry" -Detail ($query.Output -join " ")
            Write-WatchLog "SSH_RETRY rc=$($query.ExitCode)"
            if ($Once) { exit 20 }
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        $state = Get-Field $query.Output "job_state"
        $comment = Get-Field $query.Output "comment"
        $estimated = Get-Field $query.Output "estimated.start_time"
        if (-not $state) {
            $history = Invoke-KatanaSsh "qstat -x -fx $JobId 2>&1"
            $state = Get-Field $history.Output "job_state"
            $query = $history
        }

        if ($state -eq "Q" -or $state -eq "H" -or $state -eq "W") {
            Write-WatchStatus -State $state -Comment $comment -EstimatedStart $estimated
            Write-WatchLog "QUEUE state=$state estimated_start='$estimated' comment='$comment'"
        }
        elseif ($state -eq "R" -or $state -eq "E") {
            $progress = Invoke-KatanaSsh "tail -8 ${RemoteRunRoot}/logs/progress.*.tsv 2>/dev/null; for f in ${RemoteRunRoot}/results/*predictions.jsonl; do test -f \"`$f\" && printf '%s=' \"`$(basename \"`$f\")\" && wc -l < \"`$f\"; done"
            $detail = ($progress.Output -join " | ")
            Write-WatchStatus -State $state -Comment $comment -Detail $detail
            Write-WatchLog "RUN state=$state $detail"
        }
        elseif ($state -eq "F") {
            $exitStatus = Get-Field $query.Output "Exit_status"
            $ready = Invoke-KatanaSsh "test -s ${RemoteAnalysis}/archive.sha256 && test -s ${RemoteAnalysis}/confirmation_gate.json && echo ARCHIVE_READY"
            if ($exitStatus -eq "0" -and ($ready.Output -contains "ARCHIVE_READY")) {
                Write-WatchLog "DOWNLOAD_START remote=$RemoteAnalysis"
                & scp.exe -r -i $KeyPath `
                    -o HostKeyAlias=katana.restech.unsw.edu.au `
                    "${RemoteTarget}:${RemoteAnalysis}" $ResultRoot 2>&1 | ForEach-Object { Write-WatchLog "SCP $_" }
                if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }
                $localArchive = Join-Path $ResultRoot $FullJobId
                if (-not (Test-Path -LiteralPath (Join-Path $localArchive "archive.sha256"))) {
                    throw "downloaded archive is incomplete: $localArchive"
                }
                Write-WatchStatus -State "downloaded" -Detail $localArchive
                Write-WatchLog "WATCH_COMPLETE local=$localArchive"
                exit 0
            }
            Write-WatchStatus -State "failed" -Comment $comment -Detail "Exit_status=$exitStatus"
            Write-WatchLog "JOB_FAILED Exit_status=$exitStatus archive_ready=$($ready.Output -join ',')"
            exit 30
        }
        else {
            Write-WatchStatus -State "unknown" -Comment $comment -Detail ($query.Output -join " | ")
            Write-WatchLog "UNKNOWN_STATE state='$state'"
        }

        if ($Once) { exit 0 }
        Start-Sleep -Seconds $IntervalSeconds
    }
}
finally {
    if (-not $Once -and (Test-Path -LiteralPath $PidPath)) {
        $owner = (Get-Content -LiteralPath $PidPath -Raw).Trim()
        if ($owner -eq [string]$PID) {
            Remove-Item -LiteralPath $PidPath -Force
        }
    }
}
