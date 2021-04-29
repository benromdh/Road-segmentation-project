import torch
import pandas as pd
import numpy as np
from helpers  import *
from preprocessing import *

def label_patch(img, i, j, patch_size):
    """Creates the label of the patch of dimension patch_size.
    The patch will be [i*patch_size : (i+1)*patch_size, j*patche_size : (j+1)*patch_size()]
    parameters: -img: tensor of dimension (H,W)
	            -i,j: pixel from which we construct the patch
	            -patch_size: size of the patch
    """
    label = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].float().mean().item()
    
    return int(label>0.25)


def create_csv(outputs, submission_name):
    """Creates a csv submission file, with the submission_name as the file name (don't forget the .csv).
    Considering the ordering of the images, the function get_sets() in pytorch_helpers already sort them in ascending order
    parameters: -outputs: tensor with dimensions (N,C,608,608), N is the number of outputs (should be 50), C the number of colors (should be 1)
	  """
    
  # Verify if the predictions have the right size
    if ((outputs.size()[0]!=50) | (outputs.size()[1]!=1) | (outputs.size()[2]!=608) | (outputs.size()[3]!=608)):
        print('The predictions should have size (50,1,608,608)')
  
    else:
        outputs = (outputs>0.5).int()
        submission = np.array([0,0], ndmin = 2)
        N  = int(outputs.shape[2]/16)
        patch_size = 16
    for file_num in range(1,51):
        for j in range(N): # 38 is the number of 16*16 patches in a test image
            for i in range(N):
                id_ = '_'.join(['{:03d}'.format(file_num), str(j*patch_size), str(i*patch_size)])
                
                label = label_patch(outputs[file_num-1][0], i, j, patch_size)

                submission = np.concatenate((submission, np.array([id_, label], ndmin=2)))
    
    submission = np.delete(submission, 0, 0)
    submission = pd.DataFrame(submission, index = range(len(submission)), columns = ['id', 'prediction'])
    submission.to_csv(submission_name, index = False)
    
    
def create_submission(net, submission_name, device):
  """Create a csv submission directly given the network."""

  # Retrieve test images and normalize them
  Images,_ = get_sets(train=False)
  Images = data_normalization(Images)
  Images = Images.to(device)

  # Get the outputs
  outputs = torch.tensor([], device = device)
  net.eval()
  with torch.no_grad():
    for i in range(50):
      output = net(Images[i].unsqueeze(0))
      outputs = torch.cat((outputs, output))
  
  # Create the csv
  create_csv(outputs, submission_name)