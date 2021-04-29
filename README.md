# Project Road Segmentation

We provide here our solution to the Road Segmentation project of the machine learning course of EPFL. Our goal is to predict accurately the position of roads on satellite images.

The data should be divided in two directories, one containing the training images of size 400x400, and the other containing the test images of size 608x608. The dataset is available here: https://www.crowdai.org/challenges/epfl-ml-road-segmentation.

In our setup the images are located in two directories 'test_set_images' and 'training' which should be located at the same level as the code (whether it is a notebook or a python script). In those directories, the images are arranged as:
1. 'training/groundtruth/satImage_001.png'
2. 'training/images/satImage_001.png'
3. 'test_set_images/test_1/test_1.png' (Notice that in the original download from the link we have 'test_set_images/test_set_images/test_1/test_1.png').

# The network and data augmentation

We augment the dataset by cropping the four corners and the center of each images with a size of 320. We also rotate each images with angles -45, -90, 45, 90, and then crop them to a size of 320. We finally get a dataset of 900 images.
We make a train set, validation set, test set with with ratios 0.8, 0.1, 0.1 respectively.
The model we selected is a Unet with a depth of four layers and a dropout with a probability of 0.5 at the begining of each decoding layer. We trained it on the train set using the Adam optimizer with a learning rate of 2e-4, we also used the ReduceOnPlateau scheduler with a patience of 10 and a factor of 0.5.
The loss function we use is a combination of the binary cross entropy loss and the dice loss.

# The code

To be able to run the code, it is mandatory to have installed pytorch: https://pytorch.org/get-started/locally/.

IMPORTANT: If you wish to use google colab, please first add a shortcut of the shared drive to your own drive to be able to have access to the different scripts and the data.
To do so, access the drive through this link: https://drive.google.com/drive/folders/1VHxoFgvpatREXmxcpH2N89AXyhnEEMZ2?usp=sharing. Once you accessed the 'ML_Project_Final' directory, do a right click on it and select ' Add shortcut to Drive' then select 'My drive'. Now you can open the notebooks ('Final_Submission.ipynb' or 'Submission_from_net.ipynb') and run it.

We provide two ways to get the submission:
1. Train the model from scratch using the notebook 'Final_Submission.ipynb', either locally or on google colab: https://colab.research.google.com/drive/1axueu4qhYgb9-OI_MBO_gsaWFhhs5Aga?usp=sharing
2. Directly make the submission from a pre-trained model, using the 'Submission_from_net.ipynb' script, or from google colab: https://colab.research.google.com/drive/1KzxTK1NXlBZDs0sUKO5jWXxpRWhIlqgP?usp=sharing. If you choose this option, please download 'Unet_BCE_Dice_Crop320' on this link: https://drive.google.com/file/d/1OwjP6xMRrHy3UK6BhYsbFZJNAnKE2BlD/view?usp=sharing, it corresponds to our best network. Then add to your working directory. Sadly, we could not put it on github as it is too large (and we did not manage to zip it enough). Also note that you need to use GPU to load this file in the framework.



Both notebooks will output a submission file named: 'Best_Submission.csv'

Note that the network 'Unet_BCE_Dice_Crop320' (our best solution which got us our aicrowd score) we provide have been obtained using the same code in 'Final_Submission.ipynb', but some features of the networks can not be made non-deterministic (e.g. the batch normalization), so we decided to make the network available.

In the end the F1 score returned by the aicrowd platform should be 0.908 (our submission ID: 110076, username: AxelDinh).

We also used the torch-lr-finder library to get the best learning rates. We should have commented code which uses it, but in case we mistakenly left it, please install it using pip install torch_lr_finder (locally) or !pip install torch_lr_finder (on jupyter notebook).

Finally, it takes approximately 2h to train the model on google colab. (Just enough time to watch a movie).
