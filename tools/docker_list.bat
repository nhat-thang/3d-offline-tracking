@echo off
cd /d %~dp0
echo === Container (ca dang chay va da dung) === > docker_list.log
docker ps -a --format "{{.Names}} | {{.Image}} | {{.Status}} | {{.Mounts}}" >> docker_list.log 2>&1
echo. >> docker_list.log
echo === Image === >> docker_list.log
docker images --format "{{.Repository}}:{{.Tag}} | {{.Size}} | {{.CreatedSince}}" >> docker_list.log 2>&1
echo. >> docker_list.log
echo === Volume === >> docker_list.log
docker volume ls >> docker_list.log 2>&1
type docker_list.log
pause
