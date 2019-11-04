### Table of Content

-[config.py]: for defining basic configuration class;

-[mask_rcnn_coco.h]: for initializing the weights of the network training;

-[model_bak.py]: defining the four different kinds of models with backbones ["resnet30","resnet50","resnet65","resnet101"], the detailed is in the function resnet_graph,(line 168).

-[parallel_model.py]: for dealing with parallel processing of the training images.

-[steatosis.py]: for defining the steatosis config file and training the network.

-[steatosis2-Predict.py]: for predicting the steatosis segmentation with different trained model.

-[utils.py]: for drawing bounding boxes, computing iou of prediction and groundtruth, computing the recall ratio and precision.

-[visualize.py]: for visualizing the predicition and the differece with groundtruth.
