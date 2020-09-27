---
layout: post
title: PyTorch Basics
author: Maria Pantsiou
date: '2020-09-26 14:35:23 +0530'
category: AI
summary: PyTorch Basics
thumbnail: triangles_background.png

---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            displayMath: [['$$','$$']],
            inlineMath: [['$','$']],
        },
    });
</script>

# PyTorch

Time to upgrade my skills in ML, and since COVID-19 cases are rising and new restrictions are now in place, decided to do some learning while missing my social life and friends :(

## Why PyTorch?
Seems that it's like NumPy on the GPU that can parrallelize operations, so super efficient. (Also, I'm considering taking the Machine Learning Engineer Course on Udacity, and they are doing everything in PyTorch there, so win-win!) Some of the pros I've descivered are:

1. Tensor (multidimentional array) processing
2. Efficient Data Loading
3. Deep Learning Functions
4. Distributed Training
5. Provides Dynamic Computational Graphs
6. More Strong in Academia than ndustry (as TensorFlow provides additional deployment tools)

## PyTorch Data Structures
In math, the generalization of vectors and matrices to a higher dimensional space - a tensonthe generalization of vectors and matrices to a higher dimensional space - a **tensor**.

#### Tensor:
It's an entity with a defined number of dimensions called an order (**rank**).

#### Scalars:
A rank-0-tensor. Let's denote scalar value as $x∈ℝ$, where $ℝ$ is a set of real numbers.

#### Vectors:
A rank-1-tensor. Vectors belong to linear space (vector space), a set of possible vectors of a specific length. A vector consisting of real-valued scalars $x∈ℝ$ can be defined as $y∈ℝ^n$, where $y$  is vector value and $ℝ^n$ - nn-dimensional real-number vector space. $y_i$ - $i_{th}$ vector element (scalar):
$$
 y = \begin{bmatrix}
           x_{1} \\
           x_{2} \\
           \vdots \\
           x_{n}
         \end{bmatrix}
$$

#### Matrices:
A rank-2-tensor. A matrix of size $m \times n$, where $m, n \in \mathbb{N}$ (rows and columns number accordingly) consisting of real-valued scalars can be denoted as $A \in \mathbb{R}^{m \times n}$, where $\mathbb{R}^{m \times n}$ is a real-valued $m \times n$-dimensional vector space:

$$
A = \begin{bmatrix}
    x_{11}       & x_{12} & x_{13} & \dots & x_{1n} \\
    x_{21}       & x_{22} & x_{23} & \dots & x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{m1}       & x_{m2} & x_{m3} & \dots & x_{mn}
\end{bmatrix}
$$


## Code Basics
### Examples

First, we import the libraries that we need and set a random seed.

```py
import torch
import numpy as np
from matplotlib import pyplot as plt


# set seed
torch.random.manual_seed(42)
```
### PyTorch and NumPy

```py
# rank-2 (2d) tensor - float is default
torch.zeros(3, 4, dtype=torch.float16)

# rank-3 tensor
torch.zeros(2, 4, 5, dtype=torch.int16)

# rank-4 tensor
torch.rand(2, 3, 4, 5)

# create tensor from lists and np arrays
python_list = [1, 2]

# Create a numpy array from python list
numpy_array = np.array(python_list)

# Create a torch Tensor from python list
tensor_from_list = torch.tensor(python_list)

# Create a torch Tensor from Numpy array (compies memory)
tensor_from_array = torch.tensor(numpy_array)

# Another way to create a torch Tensor from Numpy array (Share same storage)
tensor_from_array_v2 = torch.from_numpy(numpy_array)

# Convert torch tensor to numpy array
array_from_tensor = tensor_from_array.numpy()

print('List:   ', python_list)
print('Array:  ', numpy_array)
print('Tensor: ', tensor_from_list)
print('Tensor: ', tensor_from_array)
print('Tensor: ', tensor_from_array_v2)
print('Array:  ', array_from_tensor)
```
```sh
Output:
Output:
List:    [1, 2]
Array:   [1 2]
Tensor:  tensor([1, 2])
Tensor:  tensor([1, 2])
Tensor:  tensor([1, 2])
Array:   [1 2]
```
### Differnce between torch.Tensor and torch.from_numpy
Pytorch aims to be an effective library for computations and avoids memory copying if it can:

```py
numpy_array[0] = 10
print('Array:  ', numpy_array)
print('Tensor: ', tensor_from_array)
print('Tensor: ', tensor_from_array_v2)
```
```sh
Output:
Output:
Array:   [10  2]
Tensor:  tensor([1, 2])
Tensor:  tensor([10,  2])
```
It also works the opposite way

### Indexing
```py
# rank-1
a = torch.rand(5)
a[2]
```
```sh
Output:
Output:
tensor(0.7936)
```
```py
# select two elements
a[[2, 4]]
```
```sh
Output:
Output:
tensor([0.7936, 0.1332])
```
```py
# select three elements with a mask
a[[True, False, False, True, True]]
```
```sh
Output:
Output:
tensor([0.6009, 0.9408, 0.1332])
```
```py
# rank-2
tensor = torch.rand((5, 3))
tensor
```
```sh
Output:
Output:
tensor([[0.2695, 0.3588, 0.1994],
       [0.5472, 0.0062, 0.9516],
       [0.0753, 0.8860, 0.5832],
       [0.3376, 0.8090, 0.5779],
       [0.9040, 0.5547, 0.3423]])
```
```py
#select row
tensor[0]
```
```sh
Output:
Output:
tensor([0.2695, 0.3588, 0.1994])
```
```py
# select element
tensor[0, 2]
```
```sh
Output:
Output:
tensor(0.1994)
```
```py
# select rows
rows = torch.tensor([0, 2, 4])
rows
```
```sh
Output:
tensor([0, 2, 4])
```
```py
tensor[rows]
```
```sh
Output:
tensor([[0.2695, 0.3588, 0.1994],
       [0.0753, 0.8860, 0.5832],
       [0.9040, 0.5547, 0.3423]])
```

### Tensor Shapes
We can reshape a tensor without the memory copying overhead. There are two methods for that: `reshape` and `view`.
The difference is the following:
1. `view` tries to return the tensor, and it shares the same memory with the original tensor. In case, if it cannot reuse the same memory due to some reasons, it just fails.
1. `reshape` always returns the tensor with the desired shape and tries to reuse the memory. If it cannot, it creates a copy.
```py
tensor = torch.rand(2, 3, 4)
tensor
```
```sh
Output:
tensor([[[0.6343, 0.3644, 0.7104, 0.9464],
        [0.7890, 0.2814, 0.7886, 0.5895],
        [0.7539, 0.1952, 0.0050, 0.3068]],

       [[0.1165, 0.9103, 0.6440, 0.7071],
        [0.6581, 0.4913, 0.8913, 0.1447],
        [0.5315, 0.1587, 0.6542, 0.3278]]])
```
```py
print('Pointer to data: ', tensor.data_ptr())
print('Shape: ', tensor.shape)
```
```
Pointer to data:  80622400
Shape:  torch.Size([2, 3, 4])
```
```py
reshaped = tensor.reshape(24)
reshaped
```
```sh
Output:
tensor([0.6343, 0.3644, 0.7104, 0.9464, 0.7890, 0.2814, 0.7886, 0.5895, 0.7539,
       0.1952, 0.0050, 0.3068, 0.1165, 0.9103, 0.6440, 0.7071, 0.6581, 0.4913,
       0.8913, 0.1447, 0.5315, 0.1587, 0.6542, 0.3278])
```
```py
view = tensor.view(3, 2, 4)
view
```
```sh
Output:
tensor([[[0.6343, 0.3644, 0.7104, 0.9464],
        [0.7890, 0.2814, 0.7886, 0.5895]],

       [[0.7539, 0.1952, 0.0050, 0.3068],
        [0.1165, 0.9103, 0.6440, 0.7071]],

       [[0.6581, 0.4913, 0.8913, 0.1447],
        [0.5315, 0.1587, 0.6542, 0.3278]]])
```
```py
# return adresses
print('Reshaped tensor - pointer to data', reshaped.data_ptr())
print('Reshaped tensor shape ', reshaped.shape)
print('Viewed tensor - pointer to data', view.data_ptr())
print('Viewed tensor shape ', view.shape)
```
```sh
Reshaped tensor - pointer to data 80622400
Reshaped tensor shape  torch.Size([24])
Viewed tensor - pointer to data 80622400
Viewed tensor shape  torch.Size([3, 2, 4])
```
```py
# assert if the original and the view tensor have the same memory adress
assert tensor.data_ptr() == view.data_ptr()
# assert is flatted tensor and reshapes are the same
assert np.all(np.equal(tensor.numpy().flat, reshaped.numpy().flat))

# print stride - the jump necessary to go from one element to the next one in the specified dimension dim
print('Original stride: ', tensor.stride())
print('Reshaped stride: ', reshaped.stride())
print('Viewed stride: ', view.stride())
```
```sh
Original stride:  (12, 4, 1)
Reshaped stride:  (1,)
Viewed stride:  (8, 4, 1)
```

If we have a multi-dimentional `tensor` and a `mask` of different dimentions we can use `expand_as` operation to create a `view` of the `mask` that has the same dimensions as the tensor we want to apply it to, but has not copied the data.

### Autograd

Pytorch supports automatic differentiation the `Autograd` module. It calculates the gradients and keeps track in forward and backward passes. For primitive tensors, you need to enable `requires_grad` flag. For advanced tensors, it is enabled by default.
```py
a = torch.rand((3, 5), requires_grad=True)
a
```
```sh
Output:
tensor([[0.3470, 0.0240, 0.7797, 0.1519, 0.7513],
       [0.7269, 0.8572, 0.1165, 0.8596, 0.2636],
       [0.6855, 0.9696, 0.4295, 0.4961, 0.3849]], requires_grad=True)
```
```py
result = a * 5
result
```
```sh
Output:
tensor([[1.7351, 0.1200, 3.8987, 0.7595, 3.7565],
       [3.6345, 4.2861, 0.5824, 4.2980, 1.3181],
       [3.4277, 4.8478, 2.1474, 2.4807, 1.9244]], grad_fn=<MulBackward0>)
```
`grad` can be implicitly created only for scalar outputs
```py
# we use sum to make it scalar to apply backward pass
mean_result = result.sum()
# Calculate Gradient
mean_result.backward()
# gradient of a
a.grad
```
```sh
Output:
tensor([[5., 5., 5., 5., 5.],
       [5., 5., 5., 5., 5.],
       [5., 5., 5., 5., 5.]])
```
We multiplied an input by 5, so as expected the calculated gradient is 5.

#### Disable autograd
We don't need to compute gradients for all the variables that are involved in the pipeline. The Pytorch API provides 2 ways to disable autograd.

1. `detach` - returns a *copy built on the same memory* of the tensor with autograd disabled. In-place size/stride/storage changes modifications are not allowed.
2. `torch.no_grad()` - It is a **context manager** that allows you to guard a series of operations from autograd without creating new tensors.

*Context managers allow you to allocate and release resources precisely when you want to. The most widely used example of context managers is the with statement.*

```py
a = torch.rand((3, 5), requires_grad=True)
detached_a = a.detach()
a
```
```sh
Output:
tensor([[0.0323, 0.7047, 0.2545, 0.3994, 0.2122],
       [0.4089, 0.1481, 0.1733, 0.6659, 0.3514],
       [0.8087, 0.3396, 0.1332, 0.4118, 0.2576]], requires_grad=True)
```
```py
detached_a
```
```sh
Output:
tensor([[0.0323, 0.7047, 0.2545, 0.3994, 0.2122],
       [0.4089, 0.1481, 0.1733, 0.6659, 0.3514],
       [0.8087, 0.3396, 0.1332, 0.4118, 0.2576]])
```
```py
detached_result = detached_a * 5
result = a * 10
```
Same as before, we cannot do backward pass that is required for autograd using multideminsional output, so we calculate the sum
```py
mean_result = result.sum()
mean_result.backward()
a.grad
```
```sh
Output:
tensor([[10., 10., 10., 10., 10.],
       [10., 10., 10., 10., 10.],
       [10., 10., 10., 10., 10.]])
```
```py
a = torch.rand((3, 5), requires_grad=True)
with torch.no_grad():
    detached_result = a * 5
result = a * 10
detached_result
```
```sh
Output:
tensor([[0.4125, 3.6998, 0.0182, 4.0520, 4.3706],
       [4.8643, 1.9103, 0.4459, 3.0621, 3.8811],
       [0.0117, 1.9325, 1.0014, 2.2813, 1.2695]])

```
```py
mean_result = result.sum()
mean_result.backward()
a.grad
```
```sh
Output:
tensor([[10., 10., 10., 10., 10.],
       [10., 10., 10., 10., 10.],
       [10., 10., 10., 10., 10.]])
```
Again, we multiplied `result` by 10, so as expected, the `grad` is 10.

Sources:

1. [OpenCV Courses](courses.opencv.org)
2. [PyTorch Docs](https://pytorch.org/docs/stable/random.html)