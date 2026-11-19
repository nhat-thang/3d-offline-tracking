@echo off
REM Xem log train theo thoi gian thuc (dong cua so khong anh huong viec train)
cd /d %~dp0
set LOG=train_refine.log
if not "%1"=="" set LOG=%1
powershell -NoProfile -Command "Get-Content %LOG% -Tail 30 -Wait"
