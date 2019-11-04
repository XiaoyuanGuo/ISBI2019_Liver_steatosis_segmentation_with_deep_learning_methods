import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#preidct steatosis segmentation

matplotlib.use('Agg')
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import utils
import visualize
from visualize import display_images
import model_bak as modellib
from model_bak import log
import numpy as np
import colorsys
import cv2
import steatosis
#%matplotlib inline 

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Dataset directory
DATASET_DIR = os.path.join("./data/", "GT_mask")
print(DATASET_DIR)


# Inference Configuration
config = steatosis.SteatosisInferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# Only inference mode is supported right now
TEST_MODE = "inference"
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return ax

# Load validation dataset
dataset = steatosis.SteatosisDataset()
dataset.load_steatosis(DATASET_DIR, "stage1_test")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


weights_path = './mask_rcnn_steatosis_0060.h5'
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)

# Or, load the last model you trained

#weights_path = model.find_last()[1]

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
for image_id in range(0,len(dataset.image_ids)):
#for image_id in range(0,2):   
    image, image_meta =modellib.load_image(dataset, config, image_id)
    info = dataset.image_info[image_id]
    # Run object detection
    results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)
    r = results[0]
    image_path= DATASET_DIR+"/stage1_test/"+dataset.image_reference(image_id)+"/masks"
    directory = os.path.dirname(image_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    alpha=0.4
    inx = -1
    invalid = 0
    N = len(r['class_ids'])
    colors =random_colors(N)

    image_masked = image
    
    for c_id in r['class_ids']:     

        inx = inx + 1
        score = r['scores'][inx] 

        pred_mask = r['masks'][:, :, inx]
        
        print("pred_mask shape is {}".format(pred_mask.shape))
        pred_mask = np.array(pred_mask, dtype=np.uint8)
        color = colors[inx]
        for c in range(3):
    np.save(image_path+'{}.npy'.format(dataset.image_reference(image_id)),r['masks'])
    cv2.imwrite(image_path+'{}_mask_Res50_Boundary_Head20_30.png'.format(dataset.image_reference(image_id)),image_masked)

print("Finished!\n")
    
