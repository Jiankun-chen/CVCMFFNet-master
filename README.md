# CVCMFF Net master
---
## About the CVCMFF Net ##
*We present a novel complex-valued convolutional and multi-feature fusion network (CVCMFF Net) specifically for building semantic segmentation of InSAR images. This CVCMFF Net not only learns from the complex-valued SAR images but also takes into account multi-scale and multi-channel feature fusion. It can effectively segment the layover, shadow, and background on both the simulated InSAR building images and the real airborne InSAR images.*

## About the Data set ##
*The simulated InSAR building dataset contains 312 simulated SAR image pairs generated from 39 different building models. Each building model is simulated at 8 viewing-angles. The sample number is 270 of the train set and is 42 of the test set. Each simulated InSAR sample contains three channels: master SAR image, slave SAR image, and interferometric phase image. This dataset serves the CVCMFF Net for building semantic segmentation of InSAR images.*

***For more information, please refer to the article: https://ieeexplore.ieee.org/document/9397870***

***Configuration of the ground truth label is: 0 = shadow; 1 = ground or single scattering; 2 = layover or secondary scattering.***

*You can get more information about the simulated InSAR building dataset from:* [https://ieee-dataport.org/documents/simulated-insar-building-dataset-cvcmff-net](https://ieee-dataport.org/documents/simulated-insar-building-dataset-cvcmff-net)

## Source file description ##


    CVCMFFNet -- _init_.py : Functions loaded into the CVCMFF Net library
              -- count_para.py: Calculation of network's FLOPs and number of parameters
              -- evaluation.py: Calculation of loss, accuarcy, and IoU
              -- GetData.py: Read input image for placeholder
              -- helpers.py: Set up and design the picture to be displayed in tensorboard
              -- inference.py: Backbone of the CVCMFF Net
              -- inputs.py: Initialize placeholder
              -- layers.py: Upsampling function
              -- training.py: Scheduling the optimizer

      Data    -- Training: train set
                 -- Images:
                    -- real1: Real channel of master SAR image
                    -- real2: Real channel of slave SAR image
                    -- imag1: Imaginary channel of master SAR image
                    -- imag2: Imaginary channel of slave SAR image
                    -- ang: Interferometric phase image
                 -- Labels: GT (one-hot)
              -- Test1: Simulated InSAR building images for test and "label" is blank.
              -- Test2: Real airborne InSAR images for test and "label" is blank.

      Output  -- model:
                 -- Training: Output tf events in train
                 -- Test: Output tf events in test
                 checkpoint files -> data-index-meta

      Outtag:   Predicted segmentation pattern

      train.py: The main function to execute CVCMFF Net (train and test)

## Requirements
***GPU MEMORY >= 8GB***

***tensorflow-gpu >= 1.14***

***imageio >= 2.6.1***

***scipy >= 1.2.1***

## Train and test ##
*Please run **train.py** and set the parameter SAVE_INTERVAL in line 34. It is the number of iteration intervals to save the model during training. When the specified SAVE_INTERVAL is reached, the network will automatically extract samples for testing and synchronously output the current network's performance on both the train set and the test set. So we can monitor the network status dynamically. We can also view the curves and predicted patterns on tensorboard.*

*The curves (orange for TRAIN and blue for TEST) of OA and LOSS versus training epoch are:
<p align="center">
  <img src=".\Curve\accuracy.svg" width=450 height=300>
  <img src=".\Curve\loss.svg" width=450 height=300>
</p>

## Experimental result ##
*The experimental curve are in the folder ./Curve: accuracy.svg; loss.svg; iou-class0.svg; iou-class1.svg; iou-class2.svg;*

*The Predicted segmentation pattern are in folders: ./outtag/result1-> simulated InSAR building images; ./outtag/result2-> real airborne InSAR images;*
 
*Segmentation results on the simulated InSAR building dataset:*
<p align="center">
  <img src=".\Data\Test1\Images\real1\22.png" width=150 height=150>
  <img src=".\Data\Test1\Images\real1\29.png" width=150 height=150>
  <img src=".\Data\Test1\Images\real1\38.png" width=150 height=150>
  <img src=".\Data\Test1\Images\real1\51.png" width=150 height=150>
</p>

<p align="center">
  <img src=".\fig\sim-22.png" width=150 height=150>
  <img src=".\fig\sim-29.png" width=150 height=150>
  <img src=".\fig\sim-38.png" width=150 height=150>
  <img src=".\fig\sim-51.png" width=150 height=150>
</p>

<p align="center">
  <img src=".\Data\Test1\Images\real1\172.png" width=150 height=150>
  <img src=".\Data\Test1\Images\real1\264.png" width=150 height=150>
  <img src=".\Data\Test1\Images\real1\270.png" width=150 height=150>
  <img src=".\Data\Test1\Images\real1\300.png" width=150 height=150>
</p>

<p align="center">
  <img src=".\fig\sim-172.png" width=150 height=150>
  <img src=".\fig\sim-264.png" width=150 height=150>
  <img src=".\fig\sim-270.png" width=150 height=150>
  <img src=".\fig\sim-300.png" width=150 height=150>
</p>

*Segmentation results on the real airborne InSAR images:*
<p align="center">
  <img src=".\Data\Test2\Images\real1\1.png" width=150 height=150>
  <img src=".\Data\Test2\Images\real1\2.png" width=150 height=150>
  <img src=".\Data\Test2\Images\real1\3.png" width=150 height=150>
  <img src=".\Data\Test2\Images\real1\4.png" width=150 height=150>
  <img src=".\Data\Test2\Images\real1\5.png" width=150 height=150>
</p>

<p align="center">
  <img src=".\fig\ra-1.png" width=150 height=150>
  <img src=".\fig\ra-2.png" width=150 height=150>
  <img src=".\fig\ra-3.png" width=150 height=150>
  <img src=".\fig\ra-4.png" width=150 height=150>
  <img src=".\fig\ra-5.png" width=150 height=150>
</p>

<p align="center">
  <img src=".\fig\ra-1-mix.png" width=150 height=150>
  <img src=".\fig\ra-2-mix.png" width=150 height=150>
  <img src=".\fig\ra-3-mix.png" width=150 height=150>
  <img src=".\fig\ra-4-mix.png" width=150 height=150>
  <img src=".\fig\ra-5-mix.png" width=150 height=150>
</p>

*The segmentation performance of CVCMFF Net is significantly improved compared with those of **other state-of-the-art networks**, as can see in:* [https://github.com/Jiankun-chen/building-semantic-segmentation-of-InSAR-images](https://github.com/Jiankun-chen/building-semantic-segmentation-of-InSAR-images)

## Feature visualization ##
*You can visualize feature maps in ***train.py*** by turning on line 50 while commenting out line 52, turning on line 110 while commenting out line 111, and turning on line 139-170.*





> Author: Jiankun Chen
> 
> This version was updated on 3/15/2021 9:30:31 PM 
