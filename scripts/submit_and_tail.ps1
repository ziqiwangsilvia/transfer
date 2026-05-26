<#
.SYNOPSIS
    Submit a catool job, wait for the pod to start, and tail its logs.
.EXAMPLE
    .\scripts\submit_and_tail.ps1 -JobsPath .\example_project -NumGpus 4
#>

param(
    [string]$JobsPath = ".\example_project",
    [int]$NumGpus = 4,
    [int]$NumNodes = 1,
    [int]$PollInterval = 10,
    [int]$MaxWait = 600
)

$env:USER_NAME = if ($env:USER_NAME) { $env:USER_NAME } else { "wangzg" }
$env:TEAM_NAME = if ($env:TEAM_NAME) { $env:TEAM_NAME } else { "cairo" }
$env:SEED = if ($env:SEED) { $env:SEED } else { "42" }

# --- Step 1: Submit the job ---
Write-Host "`n=== Submitting job ===" -ForegroundColor Cyan
catool queue --jobs-path $JobsPath --num-gpus $NumGpus --num-nodes $NumNodes --select

if ($LASTEXITCODE -ne 0) {
    Write-Host "Job submission failed." -ForegroundColor Red
    exit 1
}

# --- Step 2: Find the latest pod with our username ---
Write-Host "`n=== Finding pod ===" -ForegroundColor Cyan
Start-Sleep -Seconds 5

$podName = $null
$waited = 0

while (-not $podName -and $waited -lt 60) {
    # Get all pods sorted by creation time, find the newest one matching our username
    $raw = kubectl get pods --sort-by=.metadata.creationTimestamp -o jsonpath="{range .items[*]}{.metadata.name}{'\n'}{end}" 2>$null
    $myPods = @($raw -split "`n" | Where-Object { $_ -like "$($env:USER_NAME)-$($env:TEAM_NAME)-*-worker-0" -and $_.Trim() })
    if ($myPods.Count -gt 0) {
        $podName = $myPods[$myPods.Count - 1]
    }
    if (-not $podName) {
        Start-Sleep -Seconds 5
        $waited += 5
    }
}

if (-not $podName) {
    Write-Host "No pod found. Check 'kubectl get pods' manually." -ForegroundColor Red
    exit 1
}

# Derive job name by stripping -worker-0 / -master-0 suffix
$jobName = $podName -replace '-worker-\d+$','' -replace '-master-\d+$',''
Write-Host "Pod:  $podName" -ForegroundColor Green
Write-Host "Job:  $jobName" -ForegroundColor Green

# --- Step 3: Wait for pod to be running ---
Write-Host "`n=== Waiting for pod ===" -ForegroundColor Cyan
$waited = 0

while ($waited -lt $MaxWait) {
    $phase = (kubectl get pod $podName -o jsonpath='{.status.phase}' 2>$null)
    if ($phase -eq "Running") {
        Write-Host "Pod is running." -ForegroundColor Green
        break
    } elseif ($phase -eq "Failed" -or $phase -eq "Error" -or $phase -eq "Succeeded") {
        Write-Host "Pod ended with status: $phase" -ForegroundColor Red
        kubectl logs $podName --tail=30 2>$null
        kubectl delete pytorchjob $jobName 2>$null
        Write-Host "Cleaned up." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  Status: $phase (${waited}s)" -ForegroundColor Yellow
    Start-Sleep -Seconds $PollInterval
    $waited += $PollInterval
}

if ($waited -ge $MaxWait) {
    Write-Host "Timed out after ${MaxWait}s." -ForegroundColor Red
    exit 1
}

# --- Step 4: Tail logs ---
Write-Host "`n=== Logs ===" -ForegroundColor Cyan
kubectl logs -f $podName

# --- Step 5: Clean up ---
Write-Host "`n=== Done ===" -ForegroundColor Cyan
$phase = kubectl get pod $podName -o jsonpath='{.status.phase}' 2>$null
Write-Host "Final status: $phase"
kubectl delete pytorchjob $jobName 2>$null
Write-Host "Cleaned up." -ForegroundColor Green
