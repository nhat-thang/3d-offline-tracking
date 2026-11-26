@echo off
REM Buoc B.5: them file proto Waymo + nvitop vao moi truong CTRL (vai phut). Log: tools\waymo_9.log
cd /d %~dp0..
docker build --progress=plain -t ctrl_env2 -f waymo/Dockerfile.ctrl2 waymo > tools\waymo_9.log 2>&1
findstr /c:"proto ok" /c:"import ok" /c:"ERROR" /c:"naming to" tools\waymo_9.log
pause
