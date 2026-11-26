@echo off
REM Dang nhap Google Cloud bang DUNG tai khoan Google da dang ky Waymo Open Dataset.
REM Lam theo huong dan tren man hinh: mo link, dang nhap, dan ma xac nhan vao day. Chi can 1 lan.
docker run -it --rm -v gcloud_cfg:/root/.config/gcloud google/cloud-sdk:slim gcloud auth login --no-launch-browser
pause
