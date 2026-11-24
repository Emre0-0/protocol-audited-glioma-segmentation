param([Parameter(Mandatory)][ValidateRange(1,4)][int]$Fold,
      [Parameter(Mandatory)][string]$Project, [switch]$Resume)
$ErrorActionPreference = 'Stop'
$priority = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'powershell.exe','pwsh.exe' -and $_.CommandLine -match '-File\s+"?[^"]*queue_overlap_first\.ps1' })
if ($priority.Count) { throw 'Controlled leakage experiment has priority; V2 training deferred' }
$Run = Join-Path $PSScriptRoot 'specialist_crossfit'
$RecipePath = Join-Path $Run 'retrospective_recipe.json'
if ((Get-FileHash -LiteralPath $RecipePath).Hash.ToLowerInvariant() -ne 'f9bd9199cd488b9a2fa61c4219306bd724bbbc14536eb03fc54929eb6c82264d') { throw 'Recipe mismatch' }
$Recipe = Get-Content -LiteralPath $RecipePath -Raw | ConvertFrom-Json
$Entry = @($Recipe.inputs | Where-Object fold -eq $Fold)
if ($Entry.Count -ne 1) { throw 'Invalid starting checkpoint count' }
$Audit = Get-Content -LiteralPath (Join-Path $Run "weight_checks/fold_$Fold.json") -Raw | ConvertFrom-Json
if ($Audit.fold -ne $Fold -or $Audit.epoch -ne $Entry[0].epoch) { throw 'Weight audit mismatch' }
$Runtime = Get-Content -LiteralPath (Join-Path $Run 'runtime_check.json') -Raw | ConvertFrom-Json
foreach ($Src in $Runtime.source_hashes.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $Run "trainer_source/$($Src.Name)")).Hash -ne $Src.Value) { throw 'Trainer changed' }
}
$Dataset = 'Dataset503_BraTS2024Grouped4Ch'
$env:nnUNet_preprocessed = Join-Path $Run 'preprocessed'
$env:nnUNet_results = Join-Path $Run 'results'
$env:nnUNet_raw = Join-Path $Run 'raw'
$env:nnUNet_n_proc_DA = '4'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$SplitPath = Join-Path $env:nnUNet_preprocessed "$Dataset/splits_final.json"
if ((Get-FileHash -LiteralPath $SplitPath).Hash -ne $Recipe.sha256."processed/nnunet_preprocessed/$Dataset/splits_final.json") { throw 'Split mismatch' }
$PlanPath = Join-Path $env:nnUNet_preprocessed "$Dataset/nnUNetResEncUNet9GBPlans.json"
if ((Get-FileHash -LiteralPath $PlanPath).Hash -ne $Recipe.sha256."processed/nnunet_preprocessed/$Dataset/nnUNetResEncUNet9GBPlans.json") { throw 'Plans mismatch' }
$Splits = Get-Content -LiteralPath $SplitPath -Raw | ConvertFrom-Json
$Expected = @($Splits[$Fold].train) + @($Splits[$Fold].val)
$Done = @(Get-ChildItem -LiteralPath (Join-Path $Run 'records') -Filter '*.json' -File)
if ($Expected.Count -ne 1155 -or $Done.Count -ne 1155 -or (Compare-Object $Expected $Done.BaseName)) { throw 'Development cache incomplete' }
$Live = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'fold0_replay\.py|run_custom_trainer\.py.+train\s' -and $_.CommandLine -notmatch 'score_fold0_replay\.py' })
if ($Live.Count) { throw 'Another GPU replay or specialist training process is running' }
$Disk = Get-PSDrive -Name ([IO.Path]::GetPathRoot($Run).Substring(0,1))
if ($Disk.Free -lt 18GB) { throw '18 GiB reserve not available' }
$ModelDir = Join-Path $env:nnUNet_results "$Dataset/nnUNetTrainer_BraTSRegionAware120__nnUNetResEncUNet9GBPlans__3d_fullres/fold_$Fold"
if (Test-Path -LiteralPath (Join-Path $ModelDir 'checkpoint_final.pth')) { throw 'Fold already finished' }
if (!$Resume -and (Test-Path -LiteralPath $ModelDir)) { throw 'Existing fold output: inspect before resuming' }
$ArgsTrain = @('train','--variant','proxy','--dataset-id','503','--configuration','3d_fullres','--fold',"$Fold",'--plans','nnUNetResEncUNet9GBPlans')
if ($Resume) {
    if (!(Test-Path -LiteralPath (Join-Path $ModelDir 'checkpoint_latest.pth'))) { throw 'Latest checkpoint unavailable' }
    $ArgsTrain += '--continue-training'
} else {
    $Weights = Join-Path $Project $Entry[0].path
    $Cached = Join-Path $Run "warmstarts/fold_$Fold/checkpoint_best.pth"
    if (Test-Path -LiteralPath $Cached -PathType Leaf) { $Weights = $Cached }
    if ((Get-FileHash -LiteralPath $Weights).Hash -ne $Audit.sha256) { throw 'Warm-start file hash mismatch' }
    $ArgsTrain += @('--pretrained-weights',$Weights)
}
& py -3.10 (Join-Path $Run 'trainer_source/run_custom_trainer.py') @ArgsTrain
if ($LASTEXITCODE -ne 0) { throw "Trainer exited with $LASTEXITCODE" }
