param(
    [ValidateSet('test','benchmark','status','video')][string]$Mode='status',
    [string]$DataDir='',
    [string]$Video='',
    [string]$Output='outputs\riset_v4',
    [int[]]$Folds=@(0,2)
)
$ErrorActionPreference='Stop'
$Project=Split-Path $PSScriptRoot -Parent
Set-Location $Project
$Python=Join-Path $Project '.venv\Scripts\python.exe'
if (!(Test-Path $Python)) { $Python='python' }
if ($Mode -eq 'status') {
    Get-ChildItem (Join-Path $Project $Output) -Filter 'ringkasan_performa.csv' -Recurse -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host $_.FullName; Import-Csv $_.FullName | Format-Table -AutoSize }
    exit 0
}
if ($Mode -eq 'test') { & $Python -m unittest riset_v4.test_v4 -v; exit $LASTEXITCODE }
if ($Mode -eq 'video') {
    if (!$DataDir -or !$Video) { throw 'Isi -DataDir dan -Video dengan lokasi berkas taktis.' }
    & $Python -m riset_v4.video --data-dir $DataDir --video $Video --output $Output
    exit $LASTEXITCODE
}
$Competition=Split-Path $Project -Parent
$CacheA=Join-Path $Competition '04_Eksperimen\overnight_20261109\cache_idsse_publik'
$CacheB=Join-Path $Competition '04_Eksperimen\lanjutan_20261109\cache_public'
if (!(Test-Path $CacheA) -or !(Test-Path $CacheB)) { throw 'Cache IDSSE belum ditemukan. Gunakan python -m riset_v4.fetch_public, lalu CLI run --cache.' }
& $Python -m riset_v4.run --cache $CacheA $CacheB --output $Output --folds $Folds --epochs 5 --steps 120
exit $LASTEXITCODE
