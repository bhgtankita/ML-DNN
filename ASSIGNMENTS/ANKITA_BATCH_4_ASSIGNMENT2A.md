## $\color{Blue}Python$
### **Basic Data Types**:

**Program 1 : Numbers**

```python
eip = 3
print(type(eip)) # Prints "<class 'int'>"
print(eip)       # Prints "3"
print(eip + 1)   # Addition; prints "4"
print(eip - 1)   # Subtraction; prints "2"
print(eip * 2)   # Multiplication; prints "6"
print(eip ** 2)  # Exponentiation; prints "9"
eip += 1
print(eip)  # Prints "4"
eip *= 2
print(eip)  # Prints "8"
y = 2.5
print(type(y)) # Prints "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"
```

**Program 2 :  Booleans**

```python
eip = True
mlblr = False
print(type(eip)) # Prints "<class 'bool'>"
print(eip and mlblr) # Logical AND; prints "False"
print(eip or mlblr)  # Logical OR; prints "True"
print(not eip)   # Logical NOT; prints "False"
print(eip != mlblr)  # Logical XOR; prints "True"
```

**Program 3 : Strings**

```python
eip_in = 'hello'    # String literals can use single quotes
mlblr_in = "world"    # or double quotes; it does not matter.
print(eip_in)       # Prints "hello"
print(len(eip_in))  # String length; prints "5"
eip_out = eip_in + ' ' + mlblr_in  # String concatenation
print(eip_out)  # prints "hello world"
mlblr_out = '%s %s %d' % (eip_in, mlblr_in, 12)  # sprintf style string formatting
print(mlblr_out)  # prints "hello world 12"
```

**Program 4 : Strings**

```python
eip = "hello"
print(eip.capitalize())  # Capitalize a string; prints "Hello"
print(eip.upper())       # Convert a string to uppercase; prints "HELLO"
print(eip.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(eip.center(7))     # Center a string, padding with spaces; prints " hello "
print(eip.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"
````
---------

### **Containers**:

+ #### ***$\color{red} Lists$***: A list is the Python equivalent of an array, but is resizeable and can contain elements of different types

**Program 5 : Lists**

```python
eip = [3, 1, 2]    # Create a list
print(eip, eip[2])  # Prints "[3, 1, 2] 2"
print(eip[-1])     # Negative indices count from the end of the list; prints "2"
eip[2] = 'foo'     # Lists can contain elements of different types
print(eip)         # Prints "[3, 1, 'foo']"
eip.append('bar')  # Add a new element to the end of the list
print(eip)         # Prints "[3, 1, 'foo', 'bar']"
mlblr = eip.pop()      # Remove and return the last element of the list
print(mlblr, eip)      # Prints "bar [3, 1, 'foo']"
```

**Program 6 : Slicing**

```python
eip = list(range(5))     # range is a built-in function that creates a list of integers
print(eip)               # Prints "[0, 1, 2, 3, 4]"
print(eip[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(eip[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(eip[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(eip[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(eip[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
eip[2:4] = [8, 9]        # Assign a new sublist to a slice
print(eip)               # Prints "[0, 1, 8, 9, 4]"
```

**Program 7 : Loops**

```python
eip = ['cat', 'dog', 'monkey']
for eip in eip:
    print(eip)          # Prints "cat", "dog", "monkey", each on its own line.
```

**Program 8 : Loops**

```python
eip = ['cat', 'dog', 'monkey']
for mlblr, eip in enumerate(eip):
    print('#%d: %s' % (mlblr + 1, eip))   # Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line
```

**Program 9 : List comprehensions**

```python
eip_in = [0, 1, 2, 3, 4]
eip_out = []
for eip in eip_in:
    eip_out.append(eip ** 2)
print(eip_out)   # Prints [0, 1, 4, 9, 16]
```

**Program 10 : List comprehensions**

```python
eip_in = [0, 1, 2, 3, 4]
mlblr = [eip ** 2 for eip in eip_in]
print(mlblr)   # Prints [0, 1, 4, 9, 16]
```

**Program 11 : List comprehensions**

```python
eip_in = [0, 1, 2, 3, 4]
eip_out = [eip ** 2 for eip in eip_in if eip % 2 == 0]
print(eip_out)  # Prints "[0, 4, 16]"
```

+ #### **$\color{red}Dictionaries$**: A dictionary stores (key, value) pairs, similar to a Map in Java or an object in Javascript.

**Program 12 : Dictionaries**

```python
eip = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(eip['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in eip)     # Check if a dictionary has a given key; prints "True"
eip['fish'] = 'wet'     # Set an entry in a dictionary
print(eip['fish'])      # Prints "wet"
# print(eip['monkey'])  # KeyError: 'monkey' not a key of d
print(eip.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(eip.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del eip['fish']         # Remove an element from a dictionary
print(eip.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```

**Program 13 : Loops**

```python
eip_in = {'person': 2, 'cat': 4, 'spider': 8}
for eip in eip_in:
    mlblr = eip_in[eip]
    print('A %s has %d legs' % (eip, mlblr))    # Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

**Program 14 : Loops**

```python
eip_in = {'person': 2, 'cat': 4, 'spider': 8}
for eip, mlblr in eip_in.items():
    print('A %s has %d legs' % (eip, mlblr))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

**Program 15 : Dictionary comprehensions**

```python
eip_in = [0, 1, 2, 3, 4]
eip_out = {eip: eip ** 2 for eip in eip_in if eip % 2 == 0}
print(eip_out)  # Prints "{0: 0, 2: 4, 4: 16}"
```

+ #### **$\color{red}Sets$**: A set is an unordered collection of distinct elements.

**Program 16 : Sets**

```python
eip_in = {'cat', 'dog'}
print('cat' in eip_in)   # Check if an element is in a set; prints "True"
print('fish' in eip_in)  # prints "False"
eip_in.add('fish')       # Add an element to a set
print('fish' in eip_in)  # Prints "True"
print(len(eip_in))       # Number of elements in a set; prints "3"
eip_in.add('cat')        # Adding an element that is already in the set does nothing
print(len(eip_in))       # Prints "3"
eip_in.remove('cat')     # Remove an element from a set
print(len(eip_in))       # Prints "2"
```

**Program 17 : Loops**

```python
eip_in = {'cat', 'dog', 'fish'}
for eip, mlblr in enumerate(eip_in):
    print('#%d: %s' % (eip + 1, mlblr))  # Prints "#1: fish", "#2: dog", "#3: cat"
```

**Program 18 : Set comprehensions**

```python
from math import sqrt
eip_out = {int(sqrt(eip)) for eip in range(30)}
print(eip_out)  # Prints "{0, 1, 2, 3, 4, 5}"
```

+ #### **$\color{red}Tuples$**: A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot.

**Program 19 : Tuples**

```python
eip_in = {(eip, eip + 1): eip for eip in range(10)}  # Create a dictionary with tuple keys
eip_out = (5, 6)        # Create a tuple
print(type(eip_out))    # Prints "<class 'tuple'>"
print(eip_in[eip_out])       # Prints "5"
print(eip_in[(1, 2)])  # Prints "1"
```

### **Functions**:

**Program 20**

```python
def sign(eip):
    if eip > 0:
        return 'positive'
    elif eip < 0:
        return 'negative'
    else:
        return 'zero'

for eip in [-1, 0, 1]:
    print(sign(x))      # Prints "negative", "zero", "positive"
```

**Program 21**

```python
def hello(eip, mlblr=False):
    if eip:
        print('HELLO, %s!' % eip.upper())
    else:
        print('Hello, %s' % eip)

hello('Bob') # Prints "Hello, Bob"
hello('Fred', mlblr=True)  # Prints "HELLO, FRED!"
```

### **Classes**:

**Program 22**

```python
class Greeter(object):

    # Constructor
    def __init__(self, eip):
        self.eip = eip  # Create an instance variable

    # Instance method
    def greet(self, mlblr=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(mlblr=True)   # Call an instance method; prints "HELLO, FRED!"
```

### **Numpy**: 
Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.

+ #### **$\color{red}Arrays$**:

**Program 23**

```python
import numpy as np

eip = np.array([1, 2, 3])   # Create a rank 1 array
print(type(eip))            # Prints "<class 'numpy.ndarray'>"
print(eip.shape)            # Prints "(3,)"
print(eip[0], eip[1], eip[2])   # Prints "1 2 3"
eip[0] = 5                  # Change an element of the array
print(eip)                  # Prints "[5, 2, 3]"

mlblr = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(mlblr.shape)                     # Prints "(2, 3)"
print(mlblr[0, 0], mlblr[0, 1], mlblr[1, 0])   # Prints "1 2 4"
```

+ #### **$\color{red}Create$ $\color{red}Arrays$**:

**Program 24**

```python
import numpy as np

eip_in = np.zeros((2,2))   # Create an array of all zeros
print(eip_in)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

mlblr_in = np.ones((1,2))    # Create an array of all ones
print(mlblr_in)              # Prints "[[ 1.  1.]]"

eip = np.full((2,2), 7)  # Create a constant array
print(eip)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

mlblr = np.eye(2)         # Create a 2x2 identity matrix
print(mlblr)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

eip_out = np.random.random((2,2))  # Create an array filled with random values
print(eip_out)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
```

+ #### **$\color{red}Array$ $\color{red}Indexing$**:

**Program 25**

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
eip_in = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
eip_out = eip_in[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(eip_in[0, 1])   # Prints "2"
eip_out[0, 0] = 77     # eip_out[0, 0] is the same piece of data as a[0, 1]
print(eip_out[0, 1])   # Prints "3"
```

**Program 26**

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
eip = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
eip_in = eip[1, :]    # Rank 1 view of the second row of a
mlblr_in = eip[1:2, :]  # Rank 2 view of the second row of a
print(eip_in, eip_in.shape)  # Prints "[5 6 7 8] (4,)"
print(mlblr_in, mlblr_in.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
eip_out = eip[:, 1]
mlblr_out = eip[:, 1:2]
print(eip_out, eip_out.shape)  # Prints "[ 2  6 10] (3,)"
print(mlblr_out, mlblr_out.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
```

**Program 27**

```python
import numpy as np

eip = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(eip[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([eip[0, 0], eip[1, 1], eip[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(eip[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([eip[0, 1], eip[0, 1]]))  # Prints "[2 2]"
```

**Program 28**

```python
import numpy as np

# Create a new array from which we will select elements
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(eip)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
mlblr = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(eip[np.arange(4), mlblr])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
eip[np.arange(4), mlblr] += 10

print(eip)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])
```

**Program 29**

```python
import numpy as np

eip = np.array([[1,2], [3, 4], [5, 6]])

eip_idx = (eip > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(eip_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(eip[eip_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(eip[eip > 2])     # Prints "[3 4 5 6]"
```

**Program 30**

```python
import numpy as np

eip = np.array([1, 2])   # Let numpy choose the datatype
print(eip.dtype)         # Prints "int64"

mlblr = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(mlblr.dtype)             # Prints "float64"

eip = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(eip.dtype)                         # Prints "int64"
```

#### $\color{red}Array$ $\color{red}math$

**Program 31**

```python
import numpy as np

eip = np.array([[1,2],[3,4]], dtype=np.float64)
mlblr = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(eip + mlblr)
print(np.add(eip, mlblr))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(eip - mlblr)
print(np.subtract(eip, mlblr))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(eip * mlblr)
print(np.multiply(eip, mlblr))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(eip / mlblr)
print(np.divide(eip, mlblr))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(eip))
```

**Program 32**

```python
import numpy as np

eip = np.array([[1,2],[3,4]])
mlblr = np.array([[5,6],[7,8]])

eip_in = np.array([9,10])
mlblr_in = np.array([11, 12])

# Inner product of vectors; both produce 219
print(eip_in.dot(mlblr_in))
print(np.dot(eip_in, mlblr_in))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(eip.dot(eip_in))
print(np.dot(eip, eip_in))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(eip.dot(mlblr))
print(np.dot(eip, mlblr))
```

**Program 33**

```python
import numpy as np

eip = np.array([[1,2],[3,4]])

print(np.sum(eip))  # Compute sum of all elements; prints "10"
print(np.sum(eip, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(eip, axis=1))  # Compute sum of each row; prints "[3 7]"
```

**Program 34**

```python
import numpy as np

eip = np.array([[1,2], [3,4]])
print(eip)    # Prints "[[1 2]
            #          [3 4]]"
print(eip.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
mlblr = np.array([1,2,3])
print(mlblr)    # Prints "[1 2 3]"
print(mlblr.T)  # Prints "[1 2 3]"
```

#### $\color{red}Broadcasting$

**Program 35**

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
mlblr = np.array([1, 0, 1])
eip_out = np.empty_like(eip)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    eip_out[i, :] = eip[i, :] + mlblr

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(eip_out)
```


**Program 36**

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eip = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
mlblr = np.array([1, 0, 1])
eip_in = np.tile(mlblr, (4, 1))   # Stack 4 copies of v on top of each other
print(eip_in)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
eip_out = eip + eip_in  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

**Program 37**

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
eib = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
mlblr = np.array([1, 0, 1])
eip_out = eib + mlblr  # Add v to each row of x using broadcasting
print(eip_out)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

**Program 38**

```python
import numpy as np

# Compute outer product of vectors
eip = np.array([1,2,3])  # v has shape (3,)
mlblr = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(eip, (3, 1)) * mlblr)

# Add a vector to each row of a matrix
eip_out = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(eip_out + eip)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((eip_out.T + mlblr).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(eip_out + np.reshape(mlblr, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(eip_out * 2)
```

### **SciPy**:

#### $\color{red}Image$ $\color{red}operation$

**Program 39**

```python
from scipy.misc.pilutil import imread, imsave, imresize

# Read an JPEG image into a numpy array
eip = imread('images/cat.jpg')
print(eip.dtype, eip.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
eip_tinted = eip * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
eip_tinted = imresize(eip_tinted, (300, 300))

# Write the tinted image back to disk
imsave('images/cat_tinted.jpg', eip_tinted)
```

### **MATLAB files**: 

Distance between points

**Program 40**

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
eip = np.array([[0, 1], [1, 0], [2, 0]])
print(eip)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
mlblr = squareform(pdist(eip, 'euclidean'))
print(mlblr)
```

### **MATLAB files**: 

Plotting

**Program 41**

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
eip = np.arange(0, 3 * np.pi, 0.1)
print(eip)
mlblr = np.sin(eip)

# Plot the points using matplotlib
plt.plot(eip, mlblr)
plt.show()  # You must call plt.show() to make graphics appear.
```

**Program 42**

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
eip = np.arange(0, 3 * np.pi, 0.1)
eip_sin = np.sin(eip)
eip_cos = np.cos(eip)

# Plot the points using matplotlib
plt.plot(eip, eip_sin)
plt.plot(eip, eip_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
```

**Program 43**

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
eip = np.arange(0, 3 * np.pi, 0.1)
eip_sin = np.sin(eip)
eip_cos = np.cos(eip)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(eip, eip_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(eip, eip_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```

**Program 44**

```python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

eip = imread('images/cat.jpg')
eip_tinted = eip * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(eip)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(eip_tinted))
plt.show()
```
