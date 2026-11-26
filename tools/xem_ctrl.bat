@echo off
REM Xem truc tiep log chay CTRL (tu cap nhat). Dong cua so nay KHONG dung qua trinh chay.
cd /d %~dp0..
:cho
if exist tools\waymo_10.log goto xem
echo Chua co log. Hay chay tools\waymo_10_chay_ctrl.bat truoc. Tu kiem tra lai sau 5 giay...
timeout /t 5 /nobreak >nul
goto cho
:xem
powershell -NoProfile -ExecutionPolicy Bypass -command "Get-Content tools\waymo_10.log -Tail 25 -Wait"
pause
