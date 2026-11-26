@echo off
REM Xem truc tiep log build moi truong CTRL (tu cap nhat). Dong cua so nay KHONG dung build.
cd /d %~dp0..
powershell -command "Get-Content tools\waymo_7.log -Tail 20 -Wait"
