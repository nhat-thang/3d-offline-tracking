# Xem tien do tai goi LiDAR theo thoi gian thuc (cap nhat moi giay, tren cung 1 dong).
# Dong cua so de thoat; khong anh huong viec tai.
$dir = Join-Path $PSScriptRoot '..\data\tmp_tgz'
$sizes = @{ '01'=4.53; '02'=4.27; '03'=4.25; '04'=4.56; '05'=3.96; '06'=3.86; '07'=4.18; '08'=4.28; '09'=4.81; '10'=6.20 }
$hist = New-Object System.Collections.Queue
$lastId = ''
while ($true) {
  $f = Get-ChildItem $dir -Filter '*.tgz' -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($f) {
    $id = $f.Name.Substring(8,2)
    if ($id -ne $lastId) { $hist.Clear(); if ($lastId) { Write-Host '' }; $lastId = $id }
    $now = Get-Date; $hist.Enqueue(@($now, $f.Length)); while ($hist.Count -gt 11) { [void]$hist.Dequeue() }
    $first = $hist.Peek(); $dt = ($now - $first[0]).TotalSeconds
    if ($dt -gt 0) { $sp = ($f.Length - $first[1]) / 1MB / $dt } else { $sp = 0 }
    $mb = $f.Length / 1MB; $tot = $sizes[$id] * 1024; $pct = [math]::Min(100, 100 * $mb / $tot)
    $n = [int]($pct / 2.5); $bar = ('#' * $n) + ('.' * (40 - $n))
    if ($sp -gt 0.001) { $eta = [TimeSpan]::FromSeconds(($tot - $mb) / $sp).ToString('hh\:mm\:ss') } else { $eta = '--:--:--' }
    $line = 'goi {0} [{1}] {2,5:N1}  {3,6:N0}/{4:N0} MB  {5,5:N2} MB/s  con ~{6}' -f $id,$bar,$pct,$mb,$tot,$sp,$eta
  } else {
    $line = (Get-Date -Format 'HH:mm:ss') + '  khong co goi dang tai (dang giai nen, cho thu lai, hoac da xong)'
    $hist.Clear()
  }
  Write-Host ("`r" + $line.PadRight(100)) -NoNewline
  Start-Sleep -Milliseconds 1000
}
