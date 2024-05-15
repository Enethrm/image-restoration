import os.path #Для работы с путями к файлам и директориями.
import torch #Библиотека PyTorch для глубокого обучения.

from .utils_image import imread_uint, uint2tensor4, tensor2uint

import cv2
import numpy as np
from .network_rrdbnet import RRDBNet as net
from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse




super_resolution_router = APIRouter(prefix='/super_resolution')
@super_resolution_router.post('/')
async def super_resolution_handler(file: UploadFile):
    path_to_file = f'./temp/{file.filename}'
    with open(path_to_file, 'wb+') as file_obj:
        file_obj.write(file.file.read())

    model_path = './source/super_resolution_model/BSRGAN.pth'  # set model path | 'BSRGANx2' for scale factor 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Устройство, на котором будет выполняться модель (GPU или CPU).

    torch.cuda.empty_cache()

    # define network and load model
    sf = 4              # Масштабный коэффициент для преобразования разрешения изображения.
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()    # Модель помечается как оценочная (evaluation) и перемещается на устройство (GPU или CPU).
    model = model.to(device)
    torch.cuda.empty_cache()

    img = cv2.imread(path_to_file, cv2.IMREAD_COLOR) 


    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_L = img.unsqueeze(0)
    img_L = img_L.to(device)


    # (1) img_L
    # img_L = imread_uint(img, n_channels=3)
    # img_L = uint2tensor4(img_L)
    # img_L = img_L.to(device)

    # (2) inference
    output = model(img_L)

    # (3) img_E
    # output = tensor2uint(output)
            
    # cv2.imwrite(path_to_file, output)
    # return FileResponse(path=path_to_file)

    with torch.no_grad():
        output = model(img_L).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(path_to_file, output)
    return FileResponse(path=path_to_file)