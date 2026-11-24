param([Parameter(Mandatory)][ValidateRange(1,4)][int]$Fold)
$ErrorActionPreference = 'Stop'
$Run = Join-Path $PSScriptRoot 'specialist_crossfit'
$ds = 'Dataset503_BraTS2024Grouped4Ch'
$env:nnUNet_preprocessed = Join-Path $Run 'preprocessed'
$env:nnUNet_results = Join-Path $Run 'results'
$env:nnUNet_raw = Join-Path $Run 'raw'
$env:nnUNet_def_n_proc = '2'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$d = Join-Path $env:nnUNet_preprocessed $ds
$recipe = Join-Path $Run 'retrospective_recipe.json'
if ((Get-FileHash $recipe).Hash -ne 'f9bd9199cd488b9a2fa61c4219306bd724bbbc14536eb03fc54929eb6c82264d') { throw 'Recipe changed' }
$r = Get-Content $recipe -Raw | ConvertFrom-Json
$rt = Get-Content (Join-Path $Run 'runtime_check.json') -Raw | ConvertFrom-Json
foreach ($source in $rt.source_hashes.PSObject.Properties) {
    if ((Get-FileHash (Join-Path $Run "trainer_source/$($source.Name)")).Hash -ne $source.Value) { throw 'Specialist trainer changed' }
}
foreach ($file in 'splits_final.json','nnUNetResEncUNet9GBPlans.json') {
    if ((Get-FileHash (Join-Path $d $file)).Hash -ne $r.sha256."processed/nnunet_preprocessed/$ds/$file") { throw "Mismatch: $file" }
}
$ids = @((Get-Content (Join-Path $d 'splits_final.json') -Raw | ConvertFrom-Json)[$Fold].val)
$model = Join-Path $env:nnUNet_results "$ds/nnUNetTrainer_BraTSRegionAware120__nnUNetResEncUNet9GBPlans__3d_fullres/fold_$Fold"
foreach ($name in 'checkpoint_final.pth','checkpoint_best.pth') {
    if (!(Test-Path (Join-Path $model $name) -PathType Leaf)) { throw "Fold $Fold not ready: missing $name" }
}
$hashes = Get-Content (Join-Path $Run 'validation_reference_hashes.json') -Raw | ConvertFrom-Json
foreach ($id in $ids) {
    $name = "$id.nii.gz"
    $raw = (Get-Content (Join-Path $Run "records/$id.json") -Raw | ConvertFrom-Json).source_sha256.$name
    if (!$raw -or $raw -ne $hashes.$id -or (Get-FileHash (Join-Path $d "gt_segmentations/$name")).Hash -ne $raw) { throw "Validation reference mismatch: $id" }
}
$out = Join-Path $Run "best_validation/fold_$Fold"
if (Test-Path $out) { throw "Validation output exists: inspect before retrying: $out" }
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_custom_trainer\.py.+(train|validate)\s|(?<!score_)fold0_replay\.py' })
if ($busy.Count) { throw 'Another GPU run is active' }
$checkpoint = Join-Path $model 'checkpoint_best.pth'
$before = (Get-FileHash -LiteralPath $checkpoint).Hash
& py -3.10 (Join-Path $Run 'trainer_source/run_custom_trainer.py') validate --variant proxy --dataset-id 503 --configuration 3d_fullres --fold $Fold --plans nnUNetResEncUNet9GBPlans --checkpoint best --validation-output-root $out
if ($LASTEXITCODE -ne 0) { throw "Validation exited with $LASTEXITCODE" }
$preds = @(Get-ChildItem (Join-Path $out 'validation') -Filter '*.nii.gz' -File | ForEach-Object { $_.Name -replace '\.nii\.gz$','' })
if ($preds.Count -ne $ids.Count -or (Compare-Object $ids $preds)) { throw 'Incomplete or wrong fold predictions' }
if ((Get-FileHash -LiteralPath $checkpoint).Hash -ne $before) { throw 'Checkpoint changed during inference' }
$maskHashes = [ordered]@{}
foreach ($id in ($ids | Sort-Object)) {
    $maskHashes["$id.nii.gz"] = (Get-FileHash -LiteralPath (Join-Path $out "validation/$id.nii.gz")).Hash.ToLowerInvariant()
}
$metadata = [ordered]@{fold=$Fold; checkpoint_best_sha256=$before.ToLowerInvariant(); predictions=$maskHashes}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $out 'prediction_manifest.tmp') -Encoding utf8
Move-Item -LiteralPath (Join-Path $out 'prediction_manifest.tmp') -Destination (Join-Path $out 'prediction_manifest.json')
