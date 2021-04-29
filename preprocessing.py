import torch
import torchvision
import torchvision.transforms.functional as TF
import numpy as np

def augmentation(Images, degrees = [-45, -90, 45, 90], size = 256):
  """Augments the set of images by cropping the 4 corners and the center of each images.
  The images are also rotated with angles in degrees and then cropped.
  The crops are of size 256*256.
  parameters: -Images: set of images with size [N, 3, H, W] or [N, H, W]
              -degrees: angles to use for rotation"""

  # transformations
  FiveCrop = torchvision.transforms.FiveCrop(size)
  CenterCrop = torchvision.transforms.CenterCrop(size)

  # cropping all images in five
  FiveCrop_Images = FiveCrop(Images)

  # concatenate all 5 parts
  Augment_Images = FiveCrop_Images[0]
  Augment_Images = torch.cat((Augment_Images, FiveCrop_Images[1]))
  Augment_Images = torch.cat((Augment_Images, FiveCrop_Images[2]))
  Augment_Images = torch.cat((Augment_Images, FiveCrop_Images[3]))
  Augment_Images = torch.cat((Augment_Images, FiveCrop_Images[4]))

  # apply rotations 
  rot_Images = []
  for degree in degrees:
    rot_Images.append(CenterCrop(TF.rotate(Images, angle=int(degree))))


  # concatenate all together
  for i in range(len(rot_Images)):
    Augment_Images = torch.cat((Augment_Images, rot_Images[i]))

  return Augment_Images


def data_normalization(Images):
    """Normalizes the Images"""

    mean = torch.mean(Images, [0,2,3], keepdim = True) # obtain mean of dims = [0,2,3] and output is size 1 at these
    std = torch.std(Images, [0,2,3], keepdim = True)
    mean_l = []
    std_l = []
    for i in range(3):
      mean_l.append(mean[0][i].item())
      std_l.append(std[0][i].item())
    Normalize = torchvision.transforms.Normalize(mean_l, std_l, inplace=True)
    Images = Normalize(Images)
    return Images


def set_binary_values_GT(GroundTruths, threshold = 0.3):
  """Sets the GroundTruths as binary values"""

  GroundTruths = GroundTruths.unsqueeze(1)
  max_val = torch.max(GroundTruths)
  GroundTruths = (GroundTruths > threshold*max_val).float()
  return GroundTruths


def preprocessing(Images, GroundTruths, threshold=0.3, degrees=[90,-90,-45,45], size=256):
  """Augment and normalize the data.
  parameters: -Images: tensor of size [N, 3, H, W].
              -GroundTruths: tensor of size [N, H, W].
              -threshold: threshold on which GroundTruths values become 1.
              -degrees: angles on which images are rotated for data augmentation
              -device: device where to put the tensors ('cuda:0', or 'cpu')"""

  # Set GroundTruths to 0 and 1.
  GroundTruths = set_binary_values_GT(GroundTruths, threshold)
  
  # Augment the dataset
  Images = augmentation(Images, degrees, size)
  GroundTruths = augmentation(GroundTruths, degrees, size)
  
  # Normalize the images
  Images = data_normalization(Images)

  return Images, GroundTruths

def make_dataloaders(Images, GroundTruths, ratio_train, ratio_test, ratio_validation, batch_size = 5):
  """Makes the dataloaders, the seed have been fixed to allow reproducibility.
  parameters:   -Images: tensor of size [N,3,H,W].
                -GroundTruths: tensor of size [N,1,W,H].
                -ratio_train: float, defines the size of the dataloader for training.
                -ratio_test: float, defines the size of the dataloader for testing.
                -ratio_validation: float, defines the size of the dataloader for validation.
                -batch_size: int, size of the batches.
  returns:  -train_loader
            -test_loader
            -validation_loader"""
  
  # Get the size of the train set
  number_train = int(ratio_train*Images.size()[0])
  
  # Get the ratios for the rest of the dataset, to make we don't miss data
  sum_ratios = ratio_test + ratio_validation # e.g. 0.2
  ratio_test /= sum_ratios # e.g. 0.5
  ratio_validation /= sum_ratios # e.g. 0.5
  
  # Get the size of the test and validation tests
  rest_data = Images.size()[0] - number_train # rest of the data, e.g. 0.2 of it
  number_test = int(ratio_test*rest_data) # e.g. 0.1 of of whole data
  number_validation = rest_data - number_test

  # Create the datasets (seed fixed for reproducible results)
  dataset = torch.utils.data.TensorDataset(Images, GroundTruths)
  train_set, test_set, validation_set = torch.utils.data.dataset.random_split(dataset, lengths = [number_train, number_test, number_validation], generator=torch.Generator().manual_seed(42))

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
  validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)

  return train_loader, test_loader, validation_loader
  
  
 
def make_dataloaders2(Images, GroundTruths, ratio_train, ratio_validation, ratio_test, threshold=0.3, degrees=[90,-90,-45,45], tencrop = False, size=256, batch_size = 5):
    """Makes the dataloaders, but this time we first split the 100 original images and then augment the dataset, this way we don't biases our validation and test sets.
    Unfortunately, we did not have time to use it."""

    Images = data_normalization(Images)
    GroundTruths = set_binary_values_GT(GroundTruths, threshold)
    # Get the size of the train set
    number_train = int(ratio_train*Images.size()[0])

    # Get the ratios for the rest of the dataset, to make we don't miss data
    sum_ratios = ratio_test + ratio_validation # e.g. 0.2
    ratio_test /= sum_ratios # e.g. 0.5
    ratio_validation /= sum_ratios # e.g. 0.5

    # Get the size of the test and validation sets
    rest_data = Images.size()[0] - number_train # rest of the data, e.g. 0.2 of it
    number_test = int(ratio_test*rest_data) # e.g. 0.1 of of whole data
    number_validation = rest_data - number_test

    # Verify no set is empty
    if (number_train==0) | (number_validation==0) | (number_test==0):
      print('WARNING: Ratios not valid, a set is empty')
    
    # Get the indices of the sets
    inds = np.random.permutation(range(Images.size()[0]))
    train_inds = inds[:number_train]
    test_inds = inds[number_train:number_train+number_test]
    validation_inds = inds[number_train+number_test:]
    
    # Create the sets, and augment them
    train_set = torch.utils.data.TensorDataset(augmentation(Images[train_inds], degrees), augmentation(GroundTruths[train_inds], degrees))
    test_set = torch.utils.data.TensorDataset(augmentation(Images[test_inds], degrees), augmentation(GroundTruths[test_inds], degrees))
    validation_set = torch.utils.data.TensorDataset(augmentation(Images[validation_inds], degrees), augmentation(GroundTruths[validation_inds], degrees))
    
    # Put the sets into dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 5)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 5)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = 5)

    return train_loader, test_loader, validation_loader