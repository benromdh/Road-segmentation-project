import torch
from metrics import *
import time
import torch.nn as nn
import torch.optim as optim
from losses import *
import numpy as np
# from torch_lr_finder import LRFinder


def train(net, train_loader, validation_loader, criterion, optimizer, scheduler,  num_epochs, file_name, device, tolerance = 1e-5, use_scheduler = True):
  """ Trains the network num_epochs times."""
  print('Starting ', file_name)
  # If the the network has not been trained yet, we save it now
  if net.metrics.epochs == 0:
      torch.save({'metrics':net.metrics, 'state':net.state_dict()}, file_name)
      net.metrics.Print_Infos()
  # Make sure the network is in the right device
  net.to(device)
  
  LRS = []
  for epochs in range(num_epochs):
  
      LR = optimizer.param_groups[0]['lr'] # Just to know what is the current learning rate
      print('Epoch {} , LR {}: |'.format(net.metrics.epochs + 1, LR), end='')
      LRS.append(LR)
      
      # Initialize the epoch's metrics
      train_loss = 0
      train_accuracy = 0
      train_F1_score = 0
      train_confusion_matrix = torch.tensor([[0,0],[0,0]])

      validation_loss = 0
      validation_accuracy = 0
      validation_F1_score = 0
      validation_confusion_matrix = torch.tensor([[0,0],[0,0]])
      
      # for time monitoring
      start = time.time()
      
      net.train() # model in training mode
      for i,(imgs, gt) in enumerate(train_loader):
          # send to GPU
          imgs = imgs.to(device)
          gt = gt.to(device)

          # forward pass
          output = net(imgs)  
          loss = criterion(output, gt) 

          # backward pass and optimization
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  
          # update metrics
          train_loss += loss
          train_accuracy += compute_accuracy(output, gt)
          train_F1_score += compute_F1_score(output, gt)
          train_confusion_matrix += compute_confusion_matrix(output,gt)
          
          if i%5==0:
              print('-', end='')
      
      net.eval()
      with torch.no_grad():
          for i,(imgs, gt) in enumerate(validation_loader):
              # send to GPU
              imgs = imgs.to(device)
              gt = gt.to(device)
              # compute prediction
              output = net(imgs)

              # update the minibatches metrics
              validation_loss += criterion(output, gt)
              validation_accuracy += compute_accuracy(output, gt)
              validation_F1_score += compute_F1_score(output, gt)
              validation_confusion_matrix += compute_confusion_matrix(output, gt)

              if i%5 == 0:
                  print('-', end='')
                  
      print('|', end='')
      
      # end time and display time for 1 epoch
      end = time.time()
      print('------Epoch over in {:.3f} seconds.'.format(end-start))

      

      # We append the epoch's metrics to the network lists (we divide by the length of the dataloaders)
      net.metrics.train_losses.append(train_loss / len(train_loader))
      net.metrics.train_accuracies.append(train_accuracy/len(train_loader))
      net.metrics.train_confusion_matrices.append(train_confusion_matrix/len(train_loader))
      net.metrics.train_F1_scores.append(train_F1_score/len(train_loader))

      net.metrics.validation_losses.append(validation_loss / len(validation_loader))
      net.metrics.validation_accuracies.append(validation_accuracy/len(validation_loader))
      net.metrics.validation_confusion_matrices.append(validation_confusion_matrix/len(validation_loader))
      net.metrics.validation_F1_scores.append(validation_F1_score/len(validation_loader))
      
      net.metrics.epochs = net.metrics.epochs + 1

      net.metrics.epoch_times.append(end-start)

      # We print the current results
      net.metrics.Print_Infos()
      
      # Update the learning rate with scheduler
      if use_scheduler:
        # Make sure we use the right step for the scheduler
        if type(scheduler) == type(optim.lr_scheduler.ReduceLROnPlateau(optimizer)):
            scheduler.step(validation_loss / len(validation_loader))
        else:
            scheduler.step()
      
      # Save the Network so that we do not lose everything if colab closes
      torch.save({'metrics':net.metrics, 'state':net.state_dict()}, file_name)

      # If the net did not evolve, we break the loop
      if net.metrics.epochs > 10:
        if torch.abs(net.metrics.train_losses[-1] - net.metrics.train_losses[-2]) < tolerance:
          print('Training terminated: it converged.\n')
          break

  # We plot the evolution of the Network
  net.metrics.Plot_Infos()

def test(net, test_loader, criterion, device):
  """Computes the metrics on the test set"""

  test_loss = 0
  test_accuracy = 0
  test_F1_score = 0
  test_confusion_matrix = torch.tensor([[0,0],[0,0]])

  net = net.eval() # To evaluate, no more dropouts
  with torch.no_grad():
    for i,(imgs, gt) in enumerate(test_loader):
      imgs = imgs.to(device)
      gt = gt.to(device)
      # compute prediction
      output = net(imgs)
      # update the minibatches metrics
      test_loss += criterion(output, gt)
      test_accuracy += compute_accuracy(output, gt)
      test_F1_score += compute_F1_score(output, gt)
      test_confusion_matrix += compute_confusion_matrix(output, gt)

    net.metrics.test_losses.append(test_loss / len(test_loader))
    net.metrics.test_accuracies.append(test_accuracy/len(test_loader))
    net.metrics.test_confusion_matrices.append(test_confusion_matrix/len(test_loader))
    net.metrics.test_F1_scores.append(test_F1_score/len(test_loader))

  print('F1 score: {}'.format(net.metrics.test_F1_scores[-1]))

# We comment this section, it asks for a library which not necessary for the submissions
#def find_best_lr(net,  criterion, optimizer, train_loader, start_lr, end_lr):

#  lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
#  lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=400)
#  lr_finder.plot() # to inspect the loss-learning rate graph
#  lr_finder.reset() # to reset the model and optimizer to their initial state
  