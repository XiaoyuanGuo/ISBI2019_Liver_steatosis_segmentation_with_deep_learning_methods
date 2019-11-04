"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 steatosis.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 steatosis.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 steatosis.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 steatosis.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
from config import Config
import utils
import model_bak as modellib
import visualize

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
#DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
#RESULTS_DIR = os.path.join(ROOT_DIR, "results/steatosis/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = [
    #1
    "140_10036_20532_2000_0_0",
    #2
    "158_10425_17917_2000_976_976",
    #3
    "140_10411_5805_2000_0_0",
    #4
    "158_14195_17045_2000_976_976",
    #5
    "140_12382_8326_2000_0_0",
    #6
    "158_17033_16490_2000_976_976",
    #7
    "140_12382_8326_2000_0_0",
    #8
    "158_17420_17807_2000_976_976",
    #9
    "140_13059_7967_2000_0_0",
    #10
    "158_19340_17036_2000_976_976",
    #11
    "140_13550_9421_2000_0_0",
    #12
    "158_21135_16774_2000_976_976",
    #13
    "140_18452_13586_2000_0_0",
    #14
    "158_21495_14402_2000_976_976",
    #15
    "140_18693_12062_2000_0_0",
    #16
    "158_22666_14875_2000_976_976",
    #17
    "140_22715_11053_2000_0_0",
    #18
    "158_23234_13724_2000_976_976",
    #19
    "140_23305_11228_2000_0_0",
    #20
    "158_23642_12781_2000_976_976",
    #21
    "140_27523_10966_2000_0_0",
    #22
    "158_24661_11761_2000_976_976",
    #23
    "140_28811_19582_2000_0_0",
    #24
    "158_25317_10784_2000_976_976",
    #25
    "140_29033_23169_2000_0_0",
    #26
    "158_26727_4564_2000_976_976",
    #27
    "140_30530_9649_2000_0_0",
    #28
    "158_26757_10060_2000_976_976",
    #29
    "140_31109_21993_2000_0_0",
    #30
    "158_26957_4582_2000_976_976",
    #31
    "140_31310_27762_2000_0_0",
    #32
    "158_28434_7244_2000_976_976",
    #33
    "140_33924_9474_2000_0_0",
    #34
    "158_28434_7244_2000_976_976",
    #35
    "140_34145_10634_2000_0_0",
    #36
    "158_41349_16483_2000_976_976",
    #37
    "140_38840_7415_2000_0_0",
    #38
    "158_8637_18134_2000_976_976",
    #39
    "140_39267_6722_2000_0_0",
    #40
    "161_10714_39506_2000_976_976",
    #41
    "143_10318_33317_2000_0_0",
    #42
    "161_18109_37222_2000_976_976",
    #43
    "143_24305_22800_2000_0_0",
    #44
    "161_24872_8917_2000_976_976",
    #45
    "199_47875_66970_2000_976_976",
    
    
    
]


############################################################
#  Configurations
############################################################

class SteatosisConfig(Config):
    """Configuration for training on the steatosis segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "steatosis"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 6

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + steatosis

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (432 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between steatosis and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class SteatosisInferenceConfig(SteatosisConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1 #2
    IMAGES_PER_GPU = 1 #6
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class SteatosisDataset(utils.Dataset):

    def load_steatosis(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset steatosis, and the class steatosis
        self.add_class("steatosis", 1, "steatosis")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]

        if subset == "val":
            subset_dir = "stage1_train" 
            #if subset in ["train", "val"] else subset
            dataset_dir = os.path.join(dataset_dir, subset_dir)
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            dataset_dir = os.path.join(dataset_dir, subset)
            print('dataset_dir: {}'.format(dataset_dir))
            image_ids = next(os.walk(dataset_dir))[1]             
            if subset == "train":
                subset_dir = "stage1_train" 
               

        # Add images
        for image_id in image_ids:
            if image_id=="stage1_train":
                continue
            self.add_image(
                "steatosis",
                image_id=image_id,
                #path=os.path.join(dataset_dir, image_id))           
                path=os.path.join(dataset_dir, image_id, "image/{}.png".format(image_id)),
                mask_path = os.path.join(dataset_dir, image_id,"masks")
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mask_dir = info['mask_path']

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "steatosis":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = SteatosisDataset()
    dataset_train.load_steatosis(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SteatosisDataset()
    dataset_val.load_steatosis(dataset_dir, "val")

    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
          iaa.Fliplr(0.5),
          iaa.Flipud(0.5),
          iaa.OneOf([iaa.Affine(rotate=90),
                     iaa.Affine(rotate=180),
                     iaa.Affine(rotate=270)]),
          iaa.Multiply((0.8, 1.5)),
          iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,#20
                layers='heads')
    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=80,
                layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = SteatosisDataset()
    dataset.load_steatosis(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    config = SteatosisConfig()
    weights_path = 'mask_rcnn_coco.h5';
    ROOT_DIR = os.path.abspath("../")
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=ROOT_DIR+'/logs65')
    print ('model set up succeed')
    model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
    print('model load weight succedd')
    dataset_dir ='/labs/konglab/Xiaoyuan_Completed/Steatosis_All_in_One/data/'
    subset = "stage1_train"
    train(model,dataset_dir,subset)
   
