import argparse
import os
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data_uti import is_image_file
from model_srcnn import Net

from psnrmeter import PSNRMeter


if __name__ == "__main__":
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   # Time evaluation 
    starter.record()
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='epoch_4_50.pt', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/data/'
    images_name = [x for x in listdir(path) if is_image_file(x)]
    model = Net()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    out_path = 'results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    psnr_meter = PSNRMeter()  # Create an instance of PSNRMeter

    for image_name in tqdm(images_name, desc='convert LR images to HR images'):
    
        img = Image.open(path + image_name).convert('YCbCr')
        y, cb, cr = img.split()
        image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        if torch.cuda.is_available():
            image = image.cuda()

        out = model(image)
        out = out.cpu()
        out_img_y = out.data[0].numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        crop = transforms.CenterCrop(200)
        out_img = crop(out_img)
        out_img.save(out_path + image_name)
        
        # Calculate and print PSNR for each image
        original_img = Image.open('data/test/SRF_' + str(4) + '/target/' + image_name).convert('YCbCr')
        original_y, _, _ = original_img.split()
        out_img_y = crop(out_img_y)
        original_y = crop(original_y)
        psnr_meter.add(np.array(out_img_y)/255, np.array(original_y)/255)
        
    # Calculate and print average PSNR over all images
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    avg_psnr = psnr_meter.value()
    print(f"Average PSNR: {avg_psnr} dB")
    print(curr_time)
