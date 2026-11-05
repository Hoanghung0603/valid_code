param(
    [ValidateRange(30, 1800)]
    [int]$PollSeconds = 180,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$owner = 'tramanhofficial'
$inputPairs = 'data/kaggle_inputs/qwen4b_chunks_full_v1/qwen_chunk_pairs.parquet'
$launched = [System.Collections.Generic.HashSet[int]]::new()
$audited = [System.Collections.Generic.HashSet[int]]::new()
0..3 | ForEach-Object { [void]$launched.Add($_) }

function Get-Slug([int]$Index) {
    $partNumber = '{0:D2}' -f ($Index + 1)
    return "$owner/r2ai-qwen4b-original-chunks-full-part-$partNumber"
}

function Get-KernelState([int]$Index) {
    $slug = Get-Slug $Index
    $lines = & kaggle kernels status $slug 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to query Kaggle status for ${slug}: $($lines -join ' ')"
    }
    $state = $lines -join ' '
    Write-Host "$(Get-Date -Format o) part $($Index + 1): $state"
    return $state
}

function Receive-And-Audit([int]$Index) {
    $partNumber = '{0:D2}' -f ($Index + 1)
    $slug = Get-Slug $Index
    $auditDir = "data/audits/qwen4b_chunks_full_part${partNumber}"
    $downloadDir = Join-Path $auditDir 'download'
    $extractDir = Join-Path $auditDir 'extracted'
    $zipName = "qwen4b_chunks_full_part${partNumber}of04.zip"
    $zipPath = Join-Path $downloadDir $zipName
    $scorePath = Join-Path $extractDir 'qwen4b_chunk_scores.parquet'
    $auditPath = Join-Path $auditDir 'audit.json'
    New-Item -ItemType Directory -Force -Path $downloadDir, $extractDir | Out-Null

    if (-not (Test-Path -LiteralPath $zipPath)) {
        $lines = & kaggle kernels output $slug -p $downloadDir --file-pattern ([regex]::Escape($zipName)) --page-size 200 2>&1
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $zipPath)) {
            throw "Output download failed for part $partNumber. $($lines -join ' ')"
        }
    }
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force
    if (-not (Test-Path -LiteralPath $scorePath)) {
        throw "Missing score parquet after extracting part $partNumber"
    }
    & python scripts/audit_qwen4b_chunk_scores.py `
        --input $inputPairs `
        --scores $scorePath `
        --output $auditPath `
        --part-index $Index `
        --part-count 4
    if ($LASTEXITCODE -ne 0) {
        throw "Independent audit failed for part $partNumber"
    }
    Write-Host "Part $partNumber downloaded and independently audited."
}

function Push-NextPart([int]$Index) {
    $partNumber = '{0:D2}' -f ($Index + 1)
    $uploadDir = "data/kaggle_inputs/qwen4b_chunks_full_part${partNumber}_kernel"
    $slug = Get-Slug $Index
    $lines = & kaggle kernels push -p $uploadDir 2>&1
    $message = $lines -join "`n"
    Write-Host $message
    if ($LASTEXITCODE -ne 0 -or $message -notmatch 'successfully pushed') {
        throw "Kaggle push failed for part $partNumber"
    }
    [void]$launched.Add($Index)
    Write-Host "Submitted part $partNumber ($slug)."
}

while ($true) {
    $activeCount = 0
    foreach ($index in ($launched | Sort-Object)) {
        if ($audited.Contains($index)) { continue }
        $state = Get-KernelState $index
        if ($state -match 'COMPLETE') {
            Receive-And-Audit $index
            [void]$audited.Add($index)
        } elseif ($state -match 'ERROR|FAILED|CANCELLED|CANCELED') {
            throw "Kaggle part $($index + 1) terminated unsuccessfully: $state"
        } else {
            $activeCount++
        }
    }

    if ($Once) { break }

    while ($activeCount -lt 2 -and $launched.Count -lt 4) {
        $nextIndex = 0..3 | Where-Object { -not $launched.Contains($_) } | Select-Object -First 1
        Push-NextPart $nextIndex
        $activeCount++
    }

    if ($launched.Count -eq 4 -and $audited.Count -eq 4) {
        $scorePaths = 0..3 | ForEach-Object {
            $partNumber = '{0:D2}' -f ($_ + 1)
            "data/audits/qwen4b_chunks_full_part${partNumber}/extracted/qwen4b_chunk_scores.parquet"
        }
        $arguments = @(
            'scripts/merge_qwen4b_chunk_score_parts.py',
            '--input', $inputPairs,
            '--output-dir', 'data/audits/qwen4b_chunks_full_merged'
        )
        foreach ($scorePath in $scorePaths) {
            $arguments += @('--scores', $scorePath)
        }
        & python @arguments
        if ($LASTEXITCODE -ne 0) { throw 'Full chunk-score merge failed' }
        Write-Host 'All four Qwen4B chunk-score parts passed audit and were merged.'
        break
    }

    Write-Host "Sleeping $PollSeconds seconds before the next status check."
    Start-Sleep -Seconds $PollSeconds
}
