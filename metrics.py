import torch

def compute_accuracy(output, target):
  """Computes the accuracy of the output:
  output: tensor of size (N,C,H,W) (C should be 1 or 2 in our case), the value of output
          must be probabilities.
  target: tensor with the same dimensions as output"""

  output = (output>0.5).float()
  
  return torch.mean((output == target).float())
  
  
def compute_confusion_matrix(output, target):
  """Computes the confusion matrix of the output:
  parameters: -output: tensor of size (N,C,H,W) (C should be 1 or 2 in our case), the value of output
              must be probabilities.
              -target: tensor with the same dimensions as output"""

  output = (output>0.5).float()
  TP = torch.sum(torch.logical_and(output==1., target==1.))
  FP = torch.sum(torch.logical_and(output==1., target==0.))
  FN = torch.sum(torch.logical_and(output==0., target==1.))
  TN = torch.sum(torch.logical_and(output==0., target==0.))

  return torch.tensor([[TP, FP], [FN, TN]])


def compute_precision(output, target):
  """Computes the precision of the output:
  parameters: -output: tensor of size (N,C,H,W) (C should be 1 or 2 in our case), the value of output
              must be probabilities.
              -target: tensor with the same dimensions as output"""

  confusion_matrix = compute_confusion_matrix(output, target)
  precision = confusion_matrix[0,0] / confusion_matrix[0,:].sum() if confusion_matrix[0,:].sum() != 0.0 else 0.0

  return precision


def compute_recall(output, target):
  """Computes the recall of the output:
  parameters: -output: tensor of size (N,C,H,W) (C should be 1 or 2 in our case), the value of output
              must be probabilities.
              -target: tensor with the same dimensions as output"""

  confusion_matrix = compute_confusion_matrix(output, target)
  recall = confusion_matrix[0,0] / confusion_matrix[:,0].sum() if confusion_matrix[:,0].sum() != 0.0 else 0.0

  return recall


def compute_F1_score(output, target):
  """Computes the F1 score of the output:
  output: tensor of size (N,C,H,W) (C should be 1 or 2 in our case), the value of output
          must be probabilities.
  target: tensor with the same dimensions as output"""
  
  confusion_matrix = compute_confusion_matrix(output, target)

  precision = compute_precision(output, target)
  recall = compute_recall(output, target)

  F1_score = ((2*precision*recall)/(precision+recall)).item() if (precision+recall)!=0 else 0.0

  if (confusion_matrix[1,1] == output.size()[-1]*output.size()[-2]) | (confusion_matrix[0,0] == output.size()[-1]*output.size()[-2]):
    F1_score = 1

  return F1_score