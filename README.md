# Neural_style_transfer
Using one image to reflect its artistic style on the contents of the frames in the video stream. Implemented in Python using a pretrained VGG-19 model as a feature extractor.

## What it does?

An image is produced such that it shows the content of ine image and the featuristic style of the other image.
![Description](https://storage.googleapis.com/groundai-web-prod/media%2Fusers%2Fuser_14%2Fproject_331179%2Fimages%2Fx2.png)

## How does it work?

We use a trained model on a huge image dataset such as imagenet as a feature extractor. Here, I have used a simple VGG-19 network.</br>

The loss functions are chosen seperately for the style feature extractor as well the content feature extractor.
For the content feature extractor, we extract the feature vector from the last convolutional layers, and cost is computed by the MSE of the generated image and the input image. </br>
For the style feature extractor, we extract a number of convolutional layers from the VGG-19 model and compute something know as a **Gram Matrix**. It is the dot product of the generated image with the style image. It tells us how similar it is to the style image.</br>

By using a weighted sum of these costs, we backpropogate and change the pixel values of the image rather than changing the weights of the model.


## Using existing models with OpenCV

OpenCV's dnn module is used to load the trained pytorch models and perform inference.</br>

We are using [Johnson et al.](https://github.com/jcjohnson/fast-neural-style) trained models on different style images. This approach uses Faster Neural Style transfer suitable for performing inference on videos. </br>

Use the following keys to change the style of the frames in the video
- 'n' key for the next style
- 'p' key for the previous style
- 'q' key to quit

###### neural_style_transfer_video.py 

This program is used to run inference on custom video. </br> 
Change the **video_path** and the **models_path** in the program.</br>

###### neural_style_transfer_webcam.py

This program is used to run inference on the webcam. </br>
Change the **models_path** in the program.</br>

## Training

#### Requirements

1. Tensorflow 1.12 </br>
2. Numpy </br>
3. Matplotlib </br>
4. OpenCV </br>

###### train_style_transfer.py
To train the model from scratch with your custom style image, run this program by changing the following paths. 

* Content image path.
* Style image path.
* Path to save the result image.

You can change the hyperparameters and check out the results too!
