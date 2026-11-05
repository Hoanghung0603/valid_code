param(
    [ValidateRange(30, 1800)]
    [int]$PollSeconds = 120,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$inputPairs = 'data/kaggle_inputs/p1_reranker_full_v1/p1_pairs.parquet'
$launched = [System.Collections.Generic.HashSet[int]]::new()
$audited = [System.Collections.Generic.HashSet[int]]::new()
[void]$launched.Add(0)
[void]$launched.Add(1)

function Get-KernelState([int]$Index) {
    $slug = "ptquanh/r2ai-p1-full-qwen4b-part-$('{0:D2}' -f ($Index + 1))-2xt4"
    $lines = & kaggle kernels status $slug 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to query Kaggle status for ${slug}: $($lines -join ' ')"
    }
    $state = $lines -join ' '
    Write-Host "part $($Index + 1): $state"
    return $state
}

function Receive-And-Audit([int]$Index) {
    $partNumber = '{0:D2}' -f ($Index + 1)
    $slug = "ptquanh/r2ai-p1-full-qwen4b-part-$partNumber-2xt4"
    $downloadDir = "data/audits/p1_full_qwen4b_part${partNumber}_download"
    $zipName = "p1_full_qwen4b_part${partNumber}of04.zip"
    $zipPath = Join-Path $downloadDir $zipName
    $auditPath = "data/audits/p1_full_qwen4b_part${partNumber}/part_audit.json"
    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null

    if ((Test-Path -LiteralPath $auditPath) -and (Test-Path -LiteralPath $zipPath)) {
        $existingAudit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
        $inputHash = (Get-FileHash -LiteralPath $inputPairs -Algorithm SHA256).Hash.ToLowerInvariant()
        $zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($existingAudit.schema_version -eq 'p1_full_reranker_part_audit_v1' -and
            $existingAudit.model -eq 'qwen4b' -and
            [int]$existingAudit.part_index -eq $Index -and
            [int]$existingAudit.part_count -eq 4 -and
            $existingAudit.input_pairs_sha256 -eq $inputHash -and
            $existingAudit.part_zip_sha256 -eq $zipHash -and
            $existingAudit.lineage_exact -eq $true -and
            $existingAudit.doc_aggregation_exact -eq $true -and
            [int]$existingAudit.nonfinite_scores -eq 0) {
            Write-Host "part $partNumber already has a matching successful audit; reusing it."
            return
        }
        throw "Existing audit for part $partNumber does not match the current input/ZIP or failed an integrity check."
    }

    $pattern = [regex]::Escape($zipName)
    $lines = & kaggle kernels output $slug -p $downloadDir --file-pattern $pattern --page-size 200 2>&1
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $zipPath)) {
        throw "Output download failed for part $partNumber. $($lines -join ' ')"
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $auditPath) | Out-Null
    & python scripts/audit_p1_full_reranker_part.py `
        --input-pairs $inputPairs `
        --part $zipPath `
        --model qwen4b `
        --part-index $Index `
        --part-count 4 `
        --output $auditPath
    if ($LASTEXITCODE -ne 0) {
        throw "Independent audit failed for part $partNumber"
    }
    Write-Host "part $partNumber downloaded and independently audited: $zipPath"
}

function Push-NextPart([int]$Index) {
    $partNumber = '{0:D2}' -f ($Index + 1)
    $uploadDir = "data/kaggle_inputs/p1_full_qwen4b_part${partNumber}_kernel_upload"
    $slug = "ptquanh/r2ai-p1-full-qwen4b-part-$partNumber-2xt4"
    $lines = & kaggle kernels push -p $uploadDir --accelerator NvidiaTeslaT4 2>&1
    Write-Host ($lines -join "`n")
    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle push failed for part $partNumber"
    }
    [void]$launched.Add($Index)
    Write-Host "Submitted part $partNumber ($slug); status will be verified on the next poll."
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

    if ($launched.Count -eq 4 -and $audited.Count -eq 4) {
        $partZips = 0..3 | ForEach-Object {
            $partNumber = '{0:D2}' -f ($_ + 1)
            "data/audits/p1_full_qwen4b_part${partNumber}_download/p1_full_qwen4b_part${partNumber}of04.zip"
        }
        $mergedDir = 'data/audits/p1_full_qwen4b_merged'
        New-Item -ItemType Directory -Force -Path $mergedDir | Out-Null
        & python scripts/merge_p1_full_reranker_parts.py `
            --model qwen4b `
            --part-count 4 `
            --output "$mergedDir/p1_full_qwen4b_merged.zip" `
            @partZips
        if ($LASTEXITCODE -ne 0) { throw 'Qwen4B partition merge failed' }
        Write-Host 'All four Qwen4B parts passed independent audit and were merged.'
        break
    }

    if ($activeCount -lt 2 -and $launched.Count -lt 4) {
        $nextIndex = 0..3 | Where-Object { -not $launched.Contains($_) } | Select-Object -First 1
        Push-NextPart $nextIndex
    }

    Write-Host "Sleeping $PollSeconds seconds before the next status check."
    Start-Sleep -Seconds $PollSeconds
}
