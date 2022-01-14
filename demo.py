# These are folders
save_images = 'restored_LFs'

import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader
import imageio
from network import Net
import glob
from skimage.metrics import peak_signal_noise_ratio as PSNR
# Use MATLAB for PSNR and SSIM, using python for quick demo only.

from torch.utils.data import Dataset

shutil.rmtree(save_images, ignore_errors = True)
os.makedirs(save_images)

class load_data(Dataset):
    def __init__(self):
        self.low_lf = sorted(glob.glob('low_light_LFs_compressed/low_light*.npz'))
        self.well_lf = sorted(glob.glob('low_light_LFs_compressed/well_lit*.npz'))  
    def __len__(self):
        return len(self.low_lf)
    def __getitem__(self, idx):
        return torch.from_numpy(np.load(self.low_lf[idx])['lf']), torch.from_numpy(np.load(self.well_lf[idx])['lf'])

def collect_sai(images,r,c,dir,num,container):

  if dir=='right':    
    for i in range(num):
      images.append(container[r,c+i,:,:,:])

  if dir=='down':    
    for i in range(num):
      images.append(container[r+i,c,:,:,:])

  if dir=='left':    
    for i in range(num):
      images.append(container[r,c-i,:,:,:])

  if dir=='up':    
    for i in range(num):
      images.append(container[r-i,c,:,:,:])

  return images

def create_gifs(container,name):
  images = []
  images = collect_sai(images,0,0,'right',9, container)
  images = collect_sai(images,1,8,'down',8, container)
  images = collect_sai(images,8,7,'left',8, container)
  images = collect_sai(images,7,0,'up',7, container)
  images = collect_sai(images,1,1,'right',7, container)
  images = collect_sai(images,2,7,'down',9-3, container)
  images = collect_sai(images,7,6,'left',9-3, container)
  images = collect_sai(images,6,1,'up',9-4, container)
  images = collect_sai(images,2,2,'right',9-4, container)
  images = collect_sai(images,3,6,'down',4, container)
  images = collect_sai(images,6,5,'left',4, container)
  images = collect_sai(images,5,2,'up',3, container)
  images = collect_sai(images,3,3,'right',3, container)
  images = collect_sai(images,4,5,'down',2, container)
  images = collect_sai(images,5,4,'left',2, container)
  images = collect_sai(images,4,3,'right',2, container)
  imageio.mimsave(name, images,duration=0.005)
  return

def run_test(model, dataloader_test, save_images):
    psnr=0
    with torch.no_grad():
        model.eval()
        for image_num, img in enumerate(dataloader_test):
            print('Restoring LF ',image_num+1)
            low = img[0]#.to(next(model.parameters()).device)
            gt = img[1]#.to(next(model.parameters()).device)
            img_pred = model(low)
            pred = ((np.clip(img_pred.clone().detach().cpu().numpy(),0,1)*255).astype(np.uint8)[0]).transpose(0,1,3,4,2)
            gt = ((np.clip(gt.clone().detach().cpu().numpy(),0,1)*255).astype(np.uint8)[0]).transpose(0,1,3,4,2)
            create_gifs(pred,save_images+'/restored_LF_{}.gif'.format(image_num))
            psnr+=PSNR(gt,pred)
    print('Average PSNR = ',psnr/(image_num+1),' dB')
    return

path = os.getcwd()
print(path)
dataloader_test = DataLoader(load_data(), batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
device = torch.device("cpu")
model = Net()
checkpoint = torch.load('weights',map_location=device)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))

run_test(model, dataloader_test, save_images)


