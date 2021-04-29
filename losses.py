import torch
import torch.nn as nn

def dice_coef(output, target, smooth = 1):
   """Computes the dice coefficient of the predictions
   parameters:  -output: tensor of size (N,1,H,W), must be the probablities returned by the network.
                -target: tensor of size (N.1,H,W), the groundtruths.
                -smooth: smoothness value to prevent dividing by zero"""
   
   # flatten the labels and prediction tensors
   target_f = target.reshape(-1)
   output_f = output.reshape(-1)
   
   intersection = torch.sum(target_f * output_f)
   dice = (2. * intersection + smooth) / (torch.sum(target_f)  +  torch.sum(output_f) + smooth)
   
   return dice

class Soft_Dice_Loss(nn.Module):
  """Class for the soft dice loss"""

  def __init__(self):
    super(Soft_Dice_Loss, self).__init__()

  def forward(self, output, target):
    return 1-dice_coef(output, target)
  
class BCE_Dice_Loss(nn.Module):
  """Class for the binary cross entropy + dice loss, here we define a loss function by adding the BCELoss function and the Soft_Dice_Loss function"""

  def __init__(self):
    super(BCE_Dice_Loss, self).__init__()
    self.bce = nn.BCELoss()
    self.dice_loss = Soft_Dice_Loss()
    
  def forward(self, output, target):
    return  self.dice_loss(output,target) + self.bce(output,target)