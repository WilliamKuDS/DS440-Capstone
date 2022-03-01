import cv2
import glob
import os

color_path = "/Users/william/Downloads/initial/color"
grayscale_path = r"/Users/william/Downloads/initial/grayscale/"

for filename in glob.glob(os.path.join(color_path, '*.png')):
    print(filename)
    img=cv2.imread(filename) 
    rl=cv2.resize(img, (256,256))
    gray_image = cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(grayscale_path,filename), gray_image)