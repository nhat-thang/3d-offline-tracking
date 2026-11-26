@echo off
REM Buoc B.6b: refine CTRL co Track TTA + xoa box rong (dung GPU). Log: tools\waymo_10.log
cd /d %~dp0..
call tools\waymo_10_chay_ctrl.bat "infer_tta,empty"
