# Backpropogation in CNN


**Input image table : 1x1 image with 4 channels**
||||  x         
|--|--|--|--|
| 1  | 0 | 1 | 0 |
| 1 | 0 | 1 | 1 |
| 0 | 1 | 0 | 1 |

**Weights for hidden layer**

||||  wh         
|--|--|--|
| 0.42  | 0.88 | 0.55 |
| 0.10 | 0.73 | 0.68 |
| 0.60 | 0.18 | 0.47 |
| 0.92 | 0.11 | 0.52 |

**Biases for hidden layer**

||||  bh         
|--|--|--|
| 0.46  | 0.72 | 0.08 |

**Weights for output**

||||  wout         
|--|
| 0.30 |
| 0.25 |
| 0.23 |


**Biases for output**

||||  bout         
|--|
| 0.69 |

** Python code for backpropogation algorithm **

```python

# This program will take input images, use feed forward network and implement backpropogation to find out the correct weights and biases

import numpy as np

def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def derivatives_sigmoid(x): return x * (1 - x)             # derivative of sigmoid

# Step 1 : Initialize weights and biases with random values

# Create input array of images
X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
print("Input image Array : " )
print(X)

#Create expected output array
y = np.array([[1],[1],[0]])
print("\nExpected output Array : " )
print(y)

# Prepare weights and biases array with random values
wh = np.random.random((4,3))
print("\nWeights :")
print(wh)

#wh = np.array([[0.42,0.88,0.55],[0.10,0.73,0.68],[0.60,0.18,0.47],[0.92,0.11,0.51]])

bh = np.random.random((1, 3))
print("\nBiases :")
print(bh)

#bh = np.array([[0.46,0.72,0.08]])

# Step 2 : Calculate hidden layer input

hidden_layer_input = X.dot(wh) + bh
print("\n Hidden layer input :")
print(hidden_layer_input)

# Step 3 : Apply Sigmod on hidden layer input to generate hidden activation layer

hiddenlayer_activations = sigmoid(hidden_layer_input)
print("\n Hidden layer activations :")
print(hiddenlayer_activations)

# Step 4: Perform linear and non-linear transformation of hidden layer activation at output layer

# Prepare weights and biases array with random values

wout = np.random.random((3,1))
print("\nWeights(wout) :")
print(wout)

bout = np.random.random((1))
print("\nbiases(bout) :")
print(bout)

#wout = np.array([[0.29],[0.25],[0.23]])
#bout = np.array([0.69])

output_layer = hiddenlayer_activations.dot(wout) + bout
print("\noutput :")
print(output_layer)


output = sigmoid(output_layer )
print("\nOutput layer activations :")
print(output)

# Step 5 Calculate gradient of Error(E) at output layer
E = y - output
print("\nError :")
print(E)

#Step 6 Compute slope at output and hidden layer

Slope_output_layer = derivatives_sigmoid(output)
Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
print("\nSlope_output_layer:")
print(Slope_output_layer)
print("\nSlope_hidden_layer:")
print(Slope_hidden_layer)

# Step 7 Compute delta at output layer

d_output = E * Slope_output_layer
print("\nd_output:")
print(d_output)

# Step 8 Calculate Error at hidden layer
Error_at_hidden_layer = d_output.dot(wout.T)
print("\nError_at_hidden_layer:")
print(Error_at_hidden_layer)

#Step 9 Compute delta at hidden layer
d_hiddenlayer = Error_at_hidden_layer * Slope_hidden_layer
print("\nd_hiddenlayer:")
print(d_hiddenlayer)

# Step 10 Update weight at both output and hidden layer

wout += hiddenlayer_activations.T.dot(d_output)
wh += X.T.dot(d_hiddenlayer)

print("\nwout:")
print(wout)
print("\nwh:")
print(wh)

# Step 11 Update biases at both output and hidden layer
bout += np.sum(d_output, axis=0)
bh += np.sum(d_hiddenlayer, axis=0)
print("\nbout:")
print(bout)
print("\nbh:")
print(bh)

```

-------

## Additional details

Please refer python code samples at the below path :
[Python examples](https://github.com/bhgtankita/ML-DNN/blob/master/ASSIGNMENTS/ANKITA_BATCH_4_ASSIGNMENT2A.md)
