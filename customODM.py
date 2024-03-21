import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Flatten, MaxPool2D, Dense
from pathlib import Path

train_path = "animals-detection-images-dataset/train"
test_path = "animals-detection-images-dataset/test"

