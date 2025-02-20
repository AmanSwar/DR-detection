import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from matplotlib.colors import LinearSegmentedColormap
import torch
from torchvision import transforms
import shap


class XaiVisual:

    def __init__(self)