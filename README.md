My target was build neural network from scratch starting with simple implementation without vectorization for clear understanding about all pieces of math that is doing in neural network and in backpropagation.
I tested it on Seeds Data Set.

# Data set information
Data set is measurements of geometrical properties of kernels belonging to three different varieties of wheat. A soft X-ray technique and GRAINS package were used to construct all seven, real-valued attributes.

### File columns is:
1. area A, 
2. perimeter P, 
3. compactness C = 4*pi*A/P^2, 
4. length of kernel, 
5. width of kernel, 
6. asymmetry coefficient 
7. length of kernel groove. 
**All of these parameters were real-valued continuous.*

# Order of work
  - read and convert csv data to python list of lists
  - process feature scalling with mean normalization(for gradient descent speed up)
  - train Neural Network with 1000 iterations
  - get cross-validation error

# Results
 - Accuracy on cross-validation set is **~93%**

