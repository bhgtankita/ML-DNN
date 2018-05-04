# Convolution Foundations 

### Dilated Convolution / Atrous Convolution

+ Dilated Convolution is a convolution with dilated filter, which means it will dilating the filter before doing the usual convolution. 
+ In simple terms, dilated convolution is just a convolution applied to input with defined gaps. 
+ With this definitions, given our input is an 2D image, dilation rate k=1 is normal convolution and k=2 means skipping one pixel per input and k=4 means skipping 3 pixels. 
+ Dilated convolution is a way of increasing receptive view. With this purpose, it finds usage in applications cares more about integrating knowledge of the wider context with less cost.
+ It is useful for image segmentation, to detect the boundries of the objects in image. 
+ Dilated convolution is applied in domains beside vision as well. One good example is WaveNet text-to-speech solution and ByteNet learn time text translation. They both use dilated convolution in order to capture global view of the input with less parameters. Use them if you need a wide field of view and cannot afford multiple convolutions or larger kernels.
+ In short, dilated convolution is a simple but effective idea and you might consider it in three cases;
  1. Detection of fine-details by processing inputs in higher resolutions.
  2. Broader view of the input to capture more contextual information.
  3. Faster run-time with less parameters

A 3x3 kernel with a dilation rate of 2 will have the same field of view as a 5x5 kernel, while only using 9 parameters. Imagine taking a 5x5 kernel and deleting every second column and row.

Dilation process :

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/dilation.gif?raw=true)

Dilation rate ( defines a spacing between the values in a kernel) Samples :

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/Dilated%20Convolution.PNG?raw=true)

### Fractionally Slided or Transpose or Deconvolution

+ deconvolution is a process used to reverse the effects of convolution on recorded data. 
+ The concept of deconvolution is widely used in the techniques of signal processing and image processing.

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/conv-deconv.png?raw=true)

+ There are 2 networks over here :
  1. Encoder
  2. Decoder

Encoder Network get trained first and then output of encoder will be passed to the decoder. At each layer decoder intermediate output can be compared with the input intermediate output and we can find the loss. Once we train our decoder we can discard the encoder completely. We have to write the encoder to scale the input image. In the above image blue layers are feature maps and green layer is output image. Output of the Encoder will be an image, so it can be accepted by decoder as an input. 

+ In normal convolution you take each pixel of your input image (or feature map) and calculate the dot product with your let’s say 3x3 kernel. The number you get goes into the output pixel, and then you shift to the next pixel.

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/Normal%20Conv.gif?raw=true)

+ To do the reverse (i.e. transposed convolution) you take each pixel of your input image, you multiply each pixel of your say 3x3 kernel with the input pixel to get a 3x3 weighted kernel, then you insert this in your output image. Where the outputs overlap you sum them.

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/Transposed%20Conv.gif?raw=true)

+ Often you would use a stride larger than one to increase the upsampling effect.

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/Stride%20ge%201.gif?raw=true)

The upsampling (upsamples your image to a higher resolution) kernels can be learned just like normal convolutional kernels.

### Depthwise Convolution

This will perform a spatial convolution while keeping the channels separate and then follow with a depthwise convolution. So if we have an image with 3 channels we will seperate each channel and consider as individual image with 1 channel. As image is having 1 channel, we will pick kernel also with 1 channel. 

we traverse the 16 channels with 1 3x3 kernel each, giving us 16 feature maps. Now, before merging anything, we traverse these 16 feature maps with 32 1x1 convolutions each and only then start them add together. This results in 656 (16x3x3 + 16x32x1x1) parameters opposed to the 4608 (16x32x3x3) parameters from above.

Normal Convolution :
Input image : 100x100 having 16 channels
Convolution filter : 32x3x3x16
Output : 98x98x32
Number of parameters : 4608

Depthwise Convolution :
Input image : 100x100 (16 different images with 1 channel)
Convolution filter : 16x3x3x1 (as each image out of total 16 images is having 1 channel)
Output : 98x98x16
Stack 16 images over one another and create 1 image with 16 channels
Convolution filter : 32x1x1X16
Output : 98x98x32
Number of parameters : 16x3x3x1 + 32x1x1X16 = 656

The example is a specific implementation of a depthwise separable convolution where the so called depth multiplier is 1. This is by far the most common setup for such layers.

[Depthwise Convolution](https://github.com/bhgtankita/ML-DNN/blob/master/images/DepthWise%20Convolution.mp4)

### Group Convolution

In Group Convolution, you can define multiple paths which can execute simultaneously. It means you can have input image which can convoluted by 3x3 filter, 7x7 filter and 1x1 filter simultanously. Basically it is looking at different receptive fields simultaneously and comboning them. After that you can use 1x1 filter to reduce number of channels.

Main motivation of such convolutions is to reduce computational complexity while dividing features on groups.

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/group_convolutions.png?raw=true)


### Dropout

Its a technique for regularization which help to resolve overfitting issue. Imagine we have one layer that connects to another layer. Values that go from one layer to next layer called activations. Now take those activations and randomly for every example you train your network on, set half of them to zero. Completely randomly you are taking half of the data that is flowing through your networkand just destroy it and then randomly again. 

Your network can never rely on any given activation because they might be squashed at any given moment. So it is forced to learn a redudant representation for everything to make sure at least some of the information remains. 

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/Dropout.png?raw=true)

![](https://mlblr.com/images/dropout.gif)

-----------
### Additional Information :

**Winograd's :**
7x7 = 1x7 followed by 7x1
Error is 0.3%

**Kernel Size:**
The kernel size defines the field of view of the convolution. A common choice for 2D is 3 — that is 3x3 pixels.

**Stride:**
The stride defines the step size of the kernel when traversing the image. While its default is usually 1, we can use a stride of 2 for downsampling an image similar to MaxPooling.

**Padding:**
The padding defines how the border of a sample is handled. A (half) padded convolution will keep the spatial output dimensions equal to the input, whereas unpadded convolutions will crop away some of the borders if the kernel is larger than 1.

**Label Smoothing:**

Regularization was first proposed in 1980's.

Instead of hard labels like 0 and 1, smoothen the labels by making them close to 0 and 1

For example 0, 1 -> 0.1, 0.9