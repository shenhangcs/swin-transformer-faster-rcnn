import time
import cv2
import numpy as np
from PIL import Image
from frcnn import FRCNN


image = Image.open(r'D:\My_project\faster-rcnn-pytorch-master\img\14.jpg')
frcnn = FRCNN()
r_image = frcnn.detect_image(image)
r_image.show()