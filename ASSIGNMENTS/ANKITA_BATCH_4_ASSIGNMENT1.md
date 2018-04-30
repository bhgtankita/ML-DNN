## Terminology for CNN
------------

**$\color{Blue}Convolution$**

Convolution is the filtering process widely used in Image processing. Input image will be combined with multiple kernels/filters to extract the useful features from image like edges, corners, blobs, ridges etc. 

The image is bi directional (2D) collection of pixels in rectangular cordinates. While applying Convolution to the image, we have to select kernel matrix which is lesser than the size of input image. We will slide kernel over the complete image and along the way take the dot product between the filter and chunks of the input image. For every dot product we will get a scalar value which will form the resultant matrix. This matrix is called feature map which is the output of Convolution process.

There can be lot of destracting iformation in the image which is not relevent to what we are looking for and this is the  reason Convolution being used in image processing. 

Animated representation of Convolution :

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/5-3%20Convolution%20Small.gif?raw=true)

Example of Convolution :

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/Convolution.PNG?raw=true)

**$\color{Blue}Filters/Kernels$**

Filter/Kernel is a feature extraction technique which is used to detect important features in image. It a small matrix and each filter is having a specif task to perform like blurring, sharpening, embossing, edge detection etc. 

Filters are selected randomaly which will be trained by the network subsequently. 

**$\color{Blue}Batch$**

As you cannot pass the entire dataset into the nueral net at once. so you divide dataset into number of sets.

Batch size is total number of training examples in a single batch.

**$\color{Blue}Iterations$**

Iterations is the number of batches needed to complete one epoch.

Let's say we have 5000 training examples that we are going to use. We can divide the dataset into batches of 500 then it will take 10 iterations to complete one Epoch.

**$\color{Blue}Epochs$**

One Epoch is when an entire dataset is passed forward and backward through the neural network only once. In other terms, training the network on each item of the dataset once is an epoch. 

If you want to teach your network to recognize the letters of the alphabet, 100 epochs would mean you have 2600 individual training trials.

**$\color{Blue}1x1$ $\color{Blue}Convolution$**

1x1 convolution filter will be represented as 1x1xk, where k is number of filters which will turn into number of feature maps. This reduces the depth of the feature maps.

For example we have a 100x100x50 layer, with 50 feature maps stacked on the axis of depth. Suppose we decide to discard 30 of those feature maps and retain only 20, for reducing the complexity of computation! Here, we can deploy a 1x1 convolution. In other words, we removed 30 filters by using 1x1x20 Kernel.

It works like a ‘feature pooling’ which cuts down features. 1x1 also has lesser chance for over-fitting, due to smaller Kernel size.

Animated representation of 1x1 Convolution :

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/1x1%20Convolution.gif?raw=true)

Example of 1x1 kernel with 32 filter :

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/1X1%20Example.PNG?raw=true)

**$\color{Blue}3x3$ $\color{Blue}convolution$**

3x3 Convolutio is most commonly used for 2D input data. 

Animated representation of 3x3 Convolution :

![](https://github.com/bhgtankita/ML-DNN/blob/master/images/3x3%20Convolution.gif?raw=true)

**$\color{Blue}Feature$ $\color{Blue}Maps$**

Output of the Convolution layer is called Feature map or Activation map. In convolutional neural networks, the feature map is the output of one filter applied to the previous layer. A given filter is drawn across the entire previous layer, moved one pixel at a time. Each position results in an activation of the neuron and the output is collected in the feature map. The main logic in machine learning for doing so is to present the learning algorithm with data that it is better able to regress or classify.

**$\color{Blue}Feature$ $\color{Blue}Engineering$**

Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. If feature engineering is done correctly, it increases the predictive power of machine learning algorithms by creating features from raw data that help facilitate the machine learning process. Feature Engineering is an art.

Steps which are involved while solving any problem in machine learning are as follows:

1. Gathering data
2. Cleaning data
3. Feature engineering
4. Defining model
5. Training, testing model and predicting the output

Feature engineering is the most important art in machine learning which creates the huge difference between a good model and a bad model. Let’s see what feature engineering covers.

**$\color{Blue}Activation$ $\color{Blue}Function$**

Result of a dot product between the filter and a small chunk of the
image is a single number, i.e. activation. Activation will be transformed using an activation function, e.g., sigmoid or ReLU. Convolution results in an activation map.


![](https://github.com/bhgtankita/ML-DNN/blob/master/images/Activation%20Functions.PNG?raw=true)

**$\color{Blue}Receptive$ $\color{Blue}Field$**

In typical Nueral network each neuron is connected to a neuron in the hidden layer however when dealing with high dimension inputs like images, it is impractical to connect neurons to all neurons in the previous volume. In a CNN only small region of input layer neurons connect neurons in the hidden layer this regions are refered as receptive field.  For a single neuron it is equivalent to the size of the filter size. Each neuron has a different receptive field.

For example, input image has 32x32x3 (3 channels for RGB) and receptive feild or filter size is 5x5x3. In this case each neuron in the convolution layer will have weights to a 5x5x3 region in the input volume, for a total of 5x5x3=75 weights and 1 bias parameter. 

----------------
# Additional Details :

**How to create an account on GitHub and upload a sample project**

1. Login to your Github account. Create an account using following link if you don't have it. 
   [The world’s leading software development platform · GitHub](https://github.com/)
2. At the top right of any Github page, you should see a '+' icon. Click that, then select 'New Repository'.
3. Give your repository a name--ideally the same name as your local project. 
4. Click 'Create Repository'.

**Examples of use of MathJax in Markdown**

1. Cost Function :
$$J(\theta) = \frac{1}{2m}{\sum_{i=1}^m (h_\theta(x_i) - y_i)^2}$$

2. Gradient Descent :
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0 , \theta_1)$$

3. Normal Equation :
$$\theta = (X^TX)^{-1} X^Ty$$

4. Sigmoid Function :
$$\sigma(x) = \frac{1}{1 + e^{-a}}$$

5. Derivative of sigmoid :
$$\frac{d\sigma(x)}{d(x)} = \sigma(x) . (1 - \sigma(x))$$
