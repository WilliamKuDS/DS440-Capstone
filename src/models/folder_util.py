import shutil, os
import pandas as pd


train_images = "/Users/william/Desktop/initial/training"

#creating subfolders
for i in range(5,41):
    os.makedirs('/Users/william/Desktop/initial/training/image_' + str(i))
