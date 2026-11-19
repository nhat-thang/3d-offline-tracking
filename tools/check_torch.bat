@echo off
cd /d %~dp0
echo === Tai image PyTorch (lan dau khoang 3 GB, vai phut) === > check_torch.log
docker pull pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime >> check_torch.log 2>&1
echo. >> check_torch.log
echo === Kiem tra PyTorch thay GPU === >> check_torch.log
docker run --rm --gpus all pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0)); x=torch.randn(2048,2048,device='cuda'); print('phep nhan ma tran tren GPU OK:', float((x@x).sum())!=0); print('VRAM trong (GB):', round(torch.cuda.mem_get_info()[0]/1e9,2))" >> check_torch.log 2>&1
echo. >> check_torch.log
echo === Dung luong o C con trong === >> check_torch.log
powershell -command "Get-PSDrive C | Select-Object @{n='Free_GB';e={[math]::Round($_.Free/1GB,1)}}" >> check_torch.log 2>&1
echo === XONG === >> check_torch.log
type check_torch.log
pause
