import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

################################################################################
#### CLASS METRICS TO GIVE TO ALL NETWORKS

class metrics(nn.Module):
  """Class to keep different metrics of the network (so that we can keep them even if google colab closes).
  It also provides functions to display them.
  Metrics: -train_losses
           -test_losses
           -train_accuracies
           -test_accuracies
           -train_F1_scores
           -test_F1_scores
           -epochs
           epochs_times
  Functions: -Print_Infos()
             -Plot_Metrics()"""

  def __init__(self):

    super(metrics, self).__init__()

    self.train_losses = []
    self.train_accuracies = []
    self.train_confusion_matrices = []
    self.train_F1_scores = []
    
    self.validation_losses = []
    self.validation_accuracies = []
    self.validation_confusion_matrices = []
    self.validation_F1_scores = []
    
    self.test_losses = []
    self.test_accuracies = []
    self.test_confusion_matrices = []
    self.test_F1_scores = []



    self.epochs = len(self.train_losses)
    self.epoch_times = []

  def get_precisions(self):
    """Computes the precisions"""
    train_precisions = [(M[0,0]/M[0,:].sum()).item() if M[0,:].sum() !=0.0 else 0.0 for M in self.train_confusion_matrices]
    test_precisions = [(M[0,0]/M[0,:].sum()).item() if M[0,:].sum() !=0.0 else 0.0 for M in self.test_confusion_matrices]

    return train_precisions, test_precisions
  
  def get_recalls(self):
    """Computes the recalls"""
    train_recalls = [(M[0,0]/M[:,0].sum()).item() if M[:,0].sum() !=0.0 else 0.0 for M in self.train_confusion_matrices]
    test_recalls = [(M[0,0]/M[:,0].sum()).item() if M[:,0].sum() !=0.0 else 0.0 for M in self.test_confusion_matrices]

    return train_recalls, test_recalls
    
  def Print_Infos(self):
    """Prints the current metrics of the Network"""
    if self.epochs == 0:
      print('The network has not been trained yet.')
    else:
      print('Neural Network trained for: {} Epochs.'.format(self.epochs))
      print('Current Losses: Train = {}, Validation = {}.'.format(self.train_losses[-1], self.validation_losses[-1]))
      print('Current Accuracies: Train = {}, Validation = {}.'.format(self.train_accuracies[-1], self.validation_accuracies[-1]))
      print('Current F1 Scores: Train = {}, Validation = {}.'.format(self.train_F1_scores[-1], self.validation_F1_scores[-1]))
      print('-------------------------------------------------------------------')
  
  def Plot_Infos(self):

    """Plots the evolution of the metrics of the Network"""

    if self.epochs==0:
      print('The network has not been trained yet.')
    else:
      fig, axs = plt.subplots(2,2,figsize = (20,20), sharex=True)
      fig.suptitle('METRICS', size = 20, y = 0.92)
      axs[0,0].plot(self.train_losses, label = 'Train')
      axs[0,0].plot(self.validation_losses, label = 'Validation')
      axs[0,0].legend()
      axs[0,0].set_title('LOSSES')

      axs[0,1].plot(self.train_accuracies, label = 'Train')
      axs[0,1].plot(self.validation_accuracies, label = 'Validation')
      axs[0,1].legend()
      axs[0,1].set_title('ACCURACIES')

      axs[1,0].plot(self.train_F1_scores, label = 'Train')
      axs[1,0].plot(self.validation_F1_scores, label = 'Validation')
      axs[1,0].legend()
      axs[1,0].set_title('F1 SCORES')

      axs[1,1].plot(self.epoch_times)
      axs[1,1].set_title('EPOCH TIMES')


################################################################################
############# Networks

#####################################################
#### UNET

def doubleconv_unet(in_c, out_c, dropout= 0.0):

  d_conv = nn.Sequential(
          nn.Dropout(dropout, inplace = True), 
          nn.Conv2d(in_c, out_c, kernel_size=3, padding = 1),
          nn.BatchNorm2d(out_c),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_c, out_c, kernel_size=3, padding = 1),
          nn.BatchNorm2d(out_c),
          nn.ReLU(inplace=True))

  return d_conv

class UNet(nn.Module):

    def __init__(self, p=0.0):
        super(UNet , self).__init__()
        self.metrics = metrics()
        
        self.max_pool_2d = nn.MaxPool2d(kernel_size=2, stride=2) # kernel size of 2, stride of 2, reduces image size by factor 2
        self.double_conv_1 = doubleconv_unet(3, 64)
        self.double_conv_2 = doubleconv_unet(64, 128)
        self.double_conv_3 = doubleconv_unet(128, 256)
        self.double_conv_4 = doubleconv_unet(256, 512)
        self.double_conv_5 = doubleconv_unet(512, 1024)
        
        self.up_conv_1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
        self.double_conv_6 = doubleconv_unet(1024, 512, dropout = p)

        self.up_conv_2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
        self.double_conv_7 = doubleconv_unet(512, 256, dropout = p)
        
        self.up_conv_3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
        self.double_conv_8 = doubleconv_unet(256, 128, dropout = p)
        
        self.up_conv_4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        self.double_conv_9 = doubleconv_unet(128, 64, dropout = p)
        
        self.out = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1)

    def forward(self, image):
        # encoder
        x1 = self.double_conv_1(image) #
        x2 = self.max_pool_2d(x1)
        x3 = self.double_conv_2(x2) #
        x4 = self.max_pool_2d(x3)
        x5 = self.double_conv_3(x4) #
        x6 = self.max_pool_2d(x5)
        x7 = self.double_conv_4(x6) #
        x8 = self.max_pool_2d(x7)
        x9 = self.double_conv_5(x8) 
        
        # decoder
        x = self.up_conv_1(x9)
        x = self.double_conv_6(torch.cat([x,x7], 1))
        x = self.up_conv_2(x)
        x = self.double_conv_7(torch.cat([x,x5], 1))
        x = self.up_conv_3(x)
        x = self.double_conv_8(torch.cat([x,x3], 1))
        x = self.up_conv_4(x)
        x = self.double_conv_9(torch.cat([x,x1], 1))
        x = self.out(x) 
        x = torch.sigmoid(x)

        #print(x.size())
        return x


####################################################
#### RESNET

def convin_res(in_channels, out_channels, p):
  return  nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size = 3, padding = 1, stride=1, padding_mode='reflect'),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_channels = out_channels, out_channels= out_channels, kernel_size = 3, padding = 1, stride = 1, padding_mode='reflect'),
                      nn.Dropout(p=p,inplace=True))

def doubleconv_res(in_channels, out_channels,stride,p):
    return  nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size = 3, padding = 1, stride=stride, padding_mode='reflect'),
                        nn.Dropout(p=p,inplace=True),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels = out_channels, out_channels= out_channels, kernel_size = 3, padding = 1, stride = 1, padding_mode='reflect'))

def upconv_res(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2, stride=(2,2)))

def shortcut_conv(in_channels, out_channels, stride=1):
  return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride = stride)

class encoding_block(nn.Module):

  def __init__(self,in_channels, out_channels, p):
    super(encoding_block, self).__init__()
    self.shortcut = shortcut_conv(in_channels, out_channels, 2)
    self.doubleconv = doubleconv_res(in_channels, out_channels, 2, p)

  def forward(self, x):
    return self.doubleconv(x) + self.shortcut(x)

class decoding_block(nn.Module):
  
  def __init__(self,in_channels, out_channels, p):
    super(decoding_block, self).__init__()
    self.upconv = upconv_res(in_channels, out_channels)
    self.shortcut = shortcut_conv(in_channels, out_channels, 1)
    self.doubleconv = doubleconv_res(in_channels, out_channels, 1, p)

  def forward(self, x,y):
    x = self.upconv(x)
    x = torch.cat((y,x), dim =1)
    return self.doubleconv(x) + self.shortcut(x)

class encoder(nn.Module):

  def __init__(self,layers,p_in, p_enc):
    super(encoder,self).__init__()

    if layers > 5:
      print('Too much layers')
    self.layers = layers
      
    self.convin = convin_res(3, 64, p_in)
    self.shortcut_in = shortcut_conv(3, 64)
    self.Resblocks = nn.ModuleList()
    self.in_channels = 64

    for i in range(layers):
      self.Resblocks.append(encoding_block(self.in_channels, self.in_channels*2, p_enc))
      self.in_channels *= 2

  def forward(self, x):
    x = self.convin(x) + self.shortcut_in(x)
    residuals = [x.clone()]
    for i in range(self.layers):
      x = self.Resblocks[i](x)
      residuals.append(x.clone())
    return x, residuals

class decoder(nn.Module):

  def __init__(self,layers,p_dec):
    super(decoder, self).__init__()

    if layers > 6:
      print('Too much layers')
    self.layers = layers

    self.convout = nn.Conv2d(in_channels = 64, out_channels=1, kernel_size = 3, padding = 1)

    self.Resblocks = nn.ModuleList()
    self.in_channels = np.power(2,layers)*64

    for i in range(layers):
      self.Resblocks.append(decoding_block(self.in_channels, int(self.in_channels/2), p_dec))
      self.in_channels = int(self.in_channels/2)

  def forward(self, x, residuals):
    for i in range(len(residuals)):
      x = self.Resblocks[i](x, residuals[-(i+1)])
    return self.convout(x)

class ResNet(nn.Module):

  def __init__(self, layers, p_in, p_enc, p_dec):

    super(ResNet, self).__init__()

    self.metrics = metrics()

    if layers > 5:
      print('Too much layers')

    self.layers = layers

    self.encoder = encoder(layers, p_in, p_enc)
    self.bridge = doubleconv_res(self.encoder.in_channels, 2*self.encoder.in_channels, 2, p_enc)
    self.decoder = decoder(layers+1,p_dec)

  def forward(self,x):
  
    x,res = self.encoder(x)
    x = self.bridge(x)
    x = self.decoder(x,res)
    x = torch.sigmoid(x)
    return x