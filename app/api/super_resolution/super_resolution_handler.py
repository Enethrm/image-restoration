import os
import cv2
import torch
import numpy as np
from .RRDBNet_arch import RRDBNet

from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse


super_resolution_router = APIRouter(prefix='/super_resolution')


@super_resolution_router.post('/')
async def super_resolution_handler(file: UploadFile):
    path_to_folder = './temp'
    for file_name in os.listdir('./temp'):
        os.remove(os.path.join(path_to_folder, file_name))

    path_to_file = f'{path_to_folder}/{file.filename}'
    with open(path_to_file, 'wb+') as file_obj:
        file_obj.write(file.file.read())

    model_path = './source/super_resolution_model/RRDB_ESRGAN_x4.pth'  
    device = torch.device('cpu')

    model = RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    
    img = cv2.imread(path_to_file, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(path_to_file, output)

    return FileResponse(path=path_to_file)
