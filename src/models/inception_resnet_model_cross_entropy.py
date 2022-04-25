### Install Dependencies ###
import sys
sys.path.append("../utilities")


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import cv2
import random
import matplotlib.pyplot as plt
import pretrainedmodels
from pytorchtools import EarlyStopping
from pretrainedmodels import utils
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.notebook import tqdm
    
### Set Seed ###
random.seed(2022)
sys.path.append("../utilities")

inception = pretrainedmodels.__dict__["inceptionresnetv2"](
    num_classes=1001, 
    pretrained="imagenet+background"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Hyperparameters ###

VALIDATION_SIZE = 0.1
BATCH_SIZE = 32
EPOCHS = 33
LEARNING_RATE = 0.001
PATIENCE = 5

### Helper Functions ###

def monitor_gpu():
  """
  Monitors the GPU usage throughout the script.

  Parameters:
    desc: An optional description that can be printed along with the GPU info.

  Returns:
    The allocated and cached usage in GB.
  """
  if device.type == "cuda":
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_cached(0) / 1024**3, 1), "GB")
  else:
    print("No GPU found.")

def retrieve_training_validation_samplers(train_data, validation_size):
  """
  Obtains the training and validation indices.

  Parameters:
    train_data: The training dataset as a torchvision.datasets object.
    validation_size: The % of the dataset that will be the validation set.
  
  Returns:
    A tuple of the training and validation samplers.
  """
  num_train = len(train_data)
  indices = list(range(num_train))

  np.random.shuffle(indices)
  split = int(np.floor(validation_size * num_train))

  train_idx, valid_idx = indices[split:], indices[:split]

  return SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)

def load_checkpoint(checkpoint_path, model, optimizer):
  """
  Loads the model checkpoint to resume training.

  Parameters:
    checkpoint_path: The file location of the model checkpoint.
    model: The base model without weights loaded.
    optimizer: The optimizer without weights loaded.
  
  Returns:
    The model and optimizers with the checkpoint weights and the epoch to start.
  """
  checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint["model_state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

  return model, optimizer, checkpoint["epoch"]

def stack_lab_channels(grayscale_input, ab_input):
  """
  Stacks the L and AB channels together to create the LAB image. Then, the LAB
  image is converted to RGB.

  Parameters:
    grayscale_input: The L channel as a tensor.
    ab_input: The AB channels as a tensor.
  
  Returns:
    The RGB channels as a numpy array.
  """
  color_image = torch.cat((grayscale_input, ab_input), axis=0).numpy()
  color_image = color_image.transpose((1, 2, 0)) 

  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   

  color_image = lab2rgb(color_image.astype(np.float64))

  return color_image

def convert_to_rgb(grayscale_input, ab_input, ab_ground_truth):
  """
  Converts the grayscale, predicted, and ground truth tensors into
  image-displayable numpy arrays. 

  Parameters:
    grayscale_input: The L channel as a tensor.
    ab_input: The AB channels as a tensor.
  
  Returns:
    The grayscale, predicted, and ground truth images as RGB numpy arrays.
  """
  predicted_image = stack_lab_channels(grayscale_input, ab_input)
  ground_truth_image = stack_lab_channels(grayscale_input, ab_ground_truth)
  grayscale_input = grayscale_input.squeeze().numpy()

  return grayscale_input, predicted_image, ground_truth_image

### Model Architecture ###

class Encoder(nn.Module):
  """
  The encoder for the neural network. 
  The input shape is a 224x224x1 image, which is the L channel.
  """
  def __init__(self):
    super(Encoder, self).__init__()    

    self.input_ = nn.Conv2d(1, 64, 3, padding=1, stride=2)
    self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
    self.conv2 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
    self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
    self.conv4 = nn.Conv2d(256, 256, 3, padding=1, stride=2)
    self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
    self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
    self.conv7 = nn.Conv2d(512, 256, 3, padding=1)
  
  def forward(self, x):
    x = F.relu(self.input_(x))
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    x = F.relu(self.conv7(x))

    return x

class Decoder(nn.Module):
  """
  The decoder for the neural network. 
  The input shape is the fusion layer indicated in the paper.
  """
  def __init__(self):
    super(Decoder, self).__init__()

    self.input_1 = nn.Conv2d(1257, 256, 1)
    self.input_ = nn.Conv2d(256, 128, 3, padding=1)
    self.conv1 = nn.Conv2d(128, 64, 3, padding=1)
    self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
    self.conv4 = nn.Conv2d(32, 2, 3, padding=1)

  def forward(self, x):
    x = F.relu(self.input_1(x))
    x = F.relu(self.input_(x))
    x = F.interpolate(x, scale_factor=2)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.interpolate(x, scale_factor=2)
    x = F.relu(self.conv3(x))
    x = torch.tanh(self.conv4(x))
    x = F.interpolate(x, scale_factor=2)

    return x

inception = inception.to(device)
inception.eval()

class Network(nn.Module):
  """
  Combines the outputs of the encoder and InceptionResNetV2 model and feeds this
  fused output into the decoder to output a predicted 224x224x2 AB channel.
  """
  def __init__(self):
    super(Network, self).__init__()

    self.encoder = Encoder()
    self.encoder = self.encoder.to(device)
    
    self.decoder = Decoder()
    self.decoder = self.decoder.to(device)

  def forward(self, encoder_input, feature_input):
    encoded_img = self.encoder(encoder_input)

    with torch.no_grad():
      embedding = inception(feature_input)

    embedding = embedding.view(-1, 1001, 1, 1)

    rows = torch.cat([embedding] * 28, dim=3)
    embedding_block = torch.cat([rows] * 28, dim=2)
    fusion_block = torch.cat([encoded_img, embedding_block], dim=1)

    return self.decoder(fusion_block)

### Preprocess Data ###

# Initializing the pretrained.utils methods
load_img = utils.LoadImage()
tf_img = utils.TransformImage(inception) 

# Encoder and inception models take in different HxW images
encoder_transform = transforms.Compose([transforms.CenterCrop(224)])
inception_transform = transforms.Compose([transforms.CenterCrop(299)])

class ImageDataset(datasets.ImageFolder):
  """
  Subclass of ImageFolder that separates LAB channels into L and AB channels.
  It also transforms the image into the correctly formatted input for Inception.
  """
  def __getitem__(self, index):
    img_path, _ = self.imgs[index]

    img_inception = tf_img(inception_transform(load_img(img_path)))
    img = self.loader(img_path)

    img_original = encoder_transform(img)
    img_original = np.asarray(img_original)

    img_lab = rgb2lab(img_original)
    img_lab = (img_lab + 128) / 255
    
    img_ab = img_lab[:, :, 1:3]
    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

    img_gray = rgb2gray(img_original)
    img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

    return img_gray, img_ab, img_inception



### Create Dataloaders ###

train_data = ImageDataset("../../volume/train_data")
test_data = ImageDataset("../../volume/test_data")

train_samp, valid_samp = retrieve_training_validation_samplers(
    train_data, 
    VALIDATION_SIZE
)

train_dataloader = torch.utils.data.DataLoader(
  train_data, 
  batch_size=BATCH_SIZE, 
  sampler=train_samp,
  num_workers=0
)

valid_dataloader = torch.utils.data.DataLoader(
  train_data, 
  batch_size=BATCH_SIZE, 
  sampler=valid_samp,
  num_workers=0
)


### Train the Model ###

model = Network()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

checkpoint_path = "../model_checkpoints/colorization_model (0.002478).pt"
model, optimizer, epochs = load_checkpoint(checkpoint_path, model, optimizer)



def train_model(model, batch_size, patience, n_epochs):

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for batch, (img_gray, img_ab, img_inception) in enumerate(train_dataloader, 1):
            img_gray, img_ab, img_inception = img_gray.to(device), img_ab.to(device), img_inception.to(device)

            optimizer.zero_grad()
            output = model(img_gray, img_inception)
            loss = criterion(output, img_ab)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())


        model.eval()
        with torch.no_grad():
            for img_gray, img_ab, img_inception in valid_dataloader:
                img_gray, img_ab, img_inception = img_gray.to(device), img_ab.to(device), img_inception.to(device)

                output = model(img_gray, img_inception)
                loss = criterion(output, img_ab)
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
    
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        train_losses = []
        valid_losses = []

      
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses


model, train_loss, valid_loss = train_model(model, BATCH_SIZE, PATIENCE, EPOCHS)

### Visualize the Loss and the Early Stopping Checkpoint

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0.02, 0.03) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig('../../results/plots/cross_entropy_loss_plot.png', bbox_inches='tight')


### Predict RGB Color Values on Test Data ###


test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, num_workers=0)
img_gray, img_ab, img_inception = iter(test_dataloader).next()
img_gray, img_ab, img_inception = img_gray.to(device), img_ab.to(device), img_inception.to(device)

model.eval()
with torch.no_grad():
    output = model(img_gray, img_inception)

for idx in range(100):
  grayscale, predicted_image, ground_truth = convert_to_rgb(
      img_gray[idx].cpu(), 
      output[idx].cpu(), 
      img_ab[idx].cpu()
  )
  img = cv2.convertScaleAbs(predicted_image, alpha=(255.0))
  cv2.imwrite(f'../../results/cross_entropy_results/output_{idx}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))