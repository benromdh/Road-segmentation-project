import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_img(img):
    plt.imshow(np.array(img.permute(1,2,0).cpu().int()))

def plot_roads(gt):
    plt.imshow(gt.detach().cpu().numpy(), cmap='gray')
    
def get_sets(n = 100, train = True):
    import os
    if train:
        
        root_dir = "training/"
        image_dir = root_dir + "images/"
        gt_dir = root_dir + "groundtruth/"
        files = os.listdir(image_dir)

        Images =  torch.tensor([])
        GroundTruths = torch.tensor([])

        for i in range(n):
            image = torchvision.io.read_image(image_dir + files[i]).unsqueeze(0).float()
            groundtruth = mpimg.imread(gt_dir+files[i])
            groundtruth = torch.tensor(groundtruth).unsqueeze(0).float()
            Images = torch.cat((Images, image))
            GroundTruths = torch.cat((GroundTruths, groundtruth))
            if (i+1)%10==0:
                print('{}/{} images loaded'.format(i+1,len(files)))
                
                
    else:
        root_dir = "test_set_images/"
        image_dir = root_dir
        files = os.listdir(image_dir)
        
        import re 
        files.sort(key=lambda test_string : list( map(int, re.findall(r'\d+', test_string)))[0])
        
        Images =  torch.tensor([])
        GroundTruths = torch.tensor([])

        for i in range(len(files)):
            image = torchvision.io.read_image(image_dir + files[i] + "/" + files[i] + ".png").unsqueeze(0).float()
            Images = torch.cat((Images, image))
            if (i+1)%10==0:
                print('{}/{} images loaded'.format(i+1,len(files)))
        
    
    return Images, GroundTruths