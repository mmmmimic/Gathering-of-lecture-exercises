import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def data_generator(noise=0.1, n_samples=300, D1=True):
    # Create covariates and response variable
    if D1:
        X = np.linspace(-3, 3, num=n_samples).reshape(-1,1) # 1-D
        np.random.shuffle(X)
        y = np.random.normal((0.5*np.sin(X[:,0]*3) + X[:,0]), noise) # 1-D with trend
    else:
        X = np.random.multivariate_normal(np.zeros(3), noise*np.eye(3), size = n_samples) # 3-D
        np.random.shuffle(X)    
        y = np.sin(X[:,0]) - 5*(X[:,1]**2) + 0.5*X[:,2] # 3-D

    # Stack them together vertically to split data set
    data_set = np.vstack((X.T,y)).T
    
    train, validation, test = np.split(data_set, [int(0.35*n_samples), int(0.7*n_samples)], axis=0)
    
    # Standardization of the data, remember we do the standardization with the training set mean and standard deviation
    train_mu = np.mean(train, axis=0)
    train_sigma = np.std(train, axis=0)
    
    train = (train-train_mu)/train_sigma
    validation = (validation-train_mu)/train_sigma
    test = (test-train_mu)/train_sigma
    
    x_train, x_validation, x_test = train[:,:-1], validation[:,:-1], test[:,:-1]
    y_train, y_validation, y_test = train[:,-1], validation[:,-1], test[:,-1]

    return x_train, y_train,  x_validation, y_validation, x_test, y_test

D1 = True
x_train, y_train,  x_validation, y_validation, x_test, y_test = data_generator(noise=0.5, D1=D1)

# Initialize neural network:
# the NN is a tuple with a list with weights and list with biases
def init_NN(L):
    """
    Function that initializes our feed-forward neural network. 
    Input: 
    L: list of integers. The first element must be equal to the number of features of x and the last element 
        must be the number of outputs in the network.
    Output:
    A tuple of:
    weights: a list with randomly initialized weights of shape (in units, out units) each. The units are the ones we defined in L.
        For example, if L = [2, 3, 4] layers must be a list with a first element of shape (2, 3) and a second element of shape (3, 4). 
        The length of layers must be len(L)-1
    biases: a list with randomly initialized biases of shape (1, out_units) each. For the example above, bias would be a list of length
        2 with a first element of shape (1, 3) and a second element of shape (1, 4).
    """
    weights = []
    biases  = []
    for i in range(len(L)-1):
        weights.append(np.random.normal(loc=0.0, scale=1.0, size=[L[i],L[i+1]])) 
        biases.append(np.random.normal(loc=0.0, scale=1.0, size=[1, L[i+1]]))     
        
    return (weights, biases)

# Initialize the unit test neural network:
# Same steps as above but we will not initialize the weights randomly.
def init_NN_UT(L):
    weights = []
    biases  = []
    for i in range(len(L)-1):
        weights.append(np.ones((L[i],L[i+1]))) 
        biases.append(np.ones((1, L[i+1])))     
        
    return (weights, biases)

# Initializer the unit test neural network
L_UT  = [3, 5, 1] # neutron number:3,5,1 
NN_UT = init_NN_UT(L_UT)

# Insert code here
for layer in range(len(L_UT)-1):
    print('Layer:{}, weights:{}, biases:{}'.format(layer+1, NN_UT[layer][0], NN_UT[layer][1]))

## Glorot
import math
def init_NN_glorot_Tanh(L, uniform=False):
    """
    Initializer using the glorot initialization scheme
    """
    weights = []
    biases  = []
    for i in range(len(L)-1):
        if uniform:
            #bound = 1.0 # <- replace with proper initialization
            bound = math.sqrt(6/(L[i]+L[i+1]))
            weights.append(np.random.uniform(low=-bound, high=bound, size=[L[i],L[i+1]])) 
            biases.append(np.random.uniform(low=-bound, high=bound, size=[1, L[i+1]]))  
        else:
            #std = 1.0 # <- replace with proper initialization
            std = math.sqrt(2/(L[i]+L[i+1]))
            weights.append(np.random.normal(loc=0.0, scale=std, size=[L[i],L[i+1]])) 
            biases.append(np.random.normal(loc=0.0, scale=std, size=[1, L[i+1]]))       
        
    return (weights, biases)

## He
def init_NN_he_ReLU(L, uniform=False):
    """
    Initializer using the He initialization scheme
    """
    weights = []
    biases  = []
    for i in range(len(L)-1):
        if uniform:
            #bound = 1.0 # <- replace with proper initialization
            bound = math.sqrt(6/L[i])
            weights.append(np.random.uniform(low=-bound, high=bound, size=[L[i],L[i+1]])) 
            biases.append(np.random.uniform(low=-bound, high=bound, size=[1, L[i+1]]))  
        else:
            #std = 1.0 # <- replace with proper initialization
            std = math.sqrt(2/L[i])
            weights.append(np.random.normal(loc=0.0, scale=std, size=[L[i],L[i+1]])) 
            biases.append(np.random.normal(loc=0.0, scale=std, size=[1, L[i+1]]))       
        
    return (weights, biases)

##
# I will apply models with different initialization methods on some datasets, and compare the test error. 

def Linear(x, derivative=False):
    """
    Computes the element-wise Linear activation function for an array x
    inputs:
    x: The array where the function is applied
    derivative: if set to True will return the derivative instead of the forward pass
    """
    
    if derivative:              # Return the derivative of the function evaluated at x
        return np.ones_like(x)
    else:                       # Return the forward pass of the function at x
        return x
def Sigmoid(x, derivative=False):
    """
    Computes the element-wise Sigmoid activation function for an array x
    inputs:
    x: The array where the function is applied
    derivative: if set to True will return the derivative instead of the forward pass
    """
    f = 1/(1+np.exp(-x))
    
    if derivative:              # Return the derivative of the function evaluated at x
        return f*(1-f)
    else:                       # Return the forward pass of the function at x
        return f
def Tanh(x, derivative=False):
    """
    Computes the element-wise Sigmoid activation function for an array x
    inputs:
    x: The array where the function is applied
    derivative: if set to True will return the derivative instead of the forward pass
    """
    f = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    if derivative:              # Return the derivative of the function evaluated at x
        return 1-f**2
    else:                       # Return the forward pass of the function at x
        return f
def ReLU(x, derivative=False):
    """
    Computes the element-wise Rectifier Linear Unit activation function for an array x
    inputs:
    x: The array where the function is applied
    derivative: if set to True will return the derivative instead of the forward pass
    """
    
    if derivative:              # Return the derivative of the function evaluated at x
        return (x>0).astype(int)
    else:                       # Return the forward pass of the function at x
        return np.maximum(x, 0)
def init_NN_Glorot(L, activations, uniform=False):
    """
    Initializer using the glorot initialization scheme
    Input: 
    L:list, element number in each layer
    activations:list, activation functions
    Output:
    (weights, biases)
    """
    # Insert code here
    assert len(L) == len(activations)+1
    f'Input size error'
    coeff = {Tanh:1, Sigmoid:4, ReLU:math.sqrt(2)}
    weights = []
    biases  = []
    for i in range(len(L)-1):
        if uniform:
            bound = coeff[activations[i]]*math.sqrt(6/(L[i]+L[i+1]))
            weights.append(np.random.uniform(low=-bound, high=bound, size=[L[i],L[i+1]])) 
            biases.append(np.random.uniform(low=-bound, high=bound, size=[1, L[i+1]]))  
        else:
            std = coeff[activations[i]]*math.sqrt(2/(L[i]+L[i+1]))
            weights.append(np.random.normal(loc=0.0, scale=std, size=[L[i],L[i+1]])) 
            biases.append(np.random.normal(loc=0.0, scale=std, size=[1, L[i+1]]))       

    return (weights, biases)

# Initializes the unit test neural network
L_UT  = [3, 5, 1]
ACT_UT = [ReLU, Tanh]
NN_Glorot = init_NN_Glorot(L_UT, ACT_UT)

## how these values are calculated?
# The weird-looking factors are set in order to make the variance of the weights and biases to be 1/n_{ave}, where n_{ave} = (n_{in}+n_{out})/2 

def forward_pass(x, NN, activations):
    """
    This function performs a forward pass recursively. It saves lists for both affine transforms of units (z) and activated units (a)
    Input:
    x: The input of the network             (np.array of shape: (batch_size, number_of_features))
    NN: The initialized neural network      (tuple of list of matrices)
    activations: the activations to be used (list of functions, same len as NN)

    Output:
    a: A list of affine transformations, that is, all x*w+b.
    z: A list of activated units (ALL activated units including input and output).
    
    Shapes for the einsum:
    b: batch size
    i: size of the input hidden layer (layer l)
    o: size of the output (layer l+1)
    """
    z = [x]
    a = []
        
    for l in range(len(NN[0])):
        a.append(np.einsum('bi, io -> bo', z[l], NN[0][l]) + NN[1][l])  # The affine transform x*w+b
        z.append(activations[l](a[l]))                                  # The non-linearity    
    
    return a, z

ACT_F_UT = [Linear, Linear]
test_a, test_z = forward_pass(np.array([[1,1,1]]), NN_UT, ACT_F_UT) # input has shape (1, 3) 1 batch, 3 features

# Checking shapes consistency
assert np.all(test_z[0]==np.array([1,1,1])) # Are the input vector and the first units the same?
assert np.all(test_z[1]==test_a[0])         # Are the first affine transformations and hidden units the same?
assert np.all(test_z[2]==test_a[1])         # Are the output units and the affine transformations the same?

# Checking correctnes of values
# First layer, calculate np.sum(np.array([1,1,1])*np.array([1,1,1]))+1 = 4
assert np.all(test_z[1] == 4.)
# Second layer, calculate np.sum(np.array([4,4,4,4,4])*np.array([1,1,1,1,1]))+1 = 21
assert np.all(test_z[2] == 21.)

def squared_error(t, y, derivative=False):
    """
    Computes the squared error function and its derivative 
    Input:
    t:      target (expected output)          (np.array)
    y:      output from forward pass (np.array, must be the same shape as t)
    derivative: whether to return the derivative with respect to y or return the loss (boolean)
    """
    if np.shape(t)!=np.shape(y):
        print("t and y have different shapes")
    if derivative: # Return the derivative of the function
        return (y-t)
    else:
        return 0.5*(y-t)**2
    
def cross_entropy_loss(t, y, derivative=False):
    """
    Computes the cross entropy loss function and its derivative 
    Input:
    t:      target (expected output)          (np.array)
    y:      output from forward pass (np.array, must be the same shape as t)
    derivative: whether to return the derivative with respect to y or return the loss (boolean)
    """
    ## Insert code here
    assert t.shape == y.shape
    f't and y have different shapes'
    if derivative:
        return -t/y
    else:
        return -t*math.log(y)

def backward_pass(x, t, y, z, a, NN, activations, loss):
    """
    This function performs a backward pass ITERATIVELY. It saves lists all of the derivatives in the process
    
    Input:
    x:           The input used for the batch                (np.array)
    t:           The observed targets                        (np.array, the first dimension must be the same to x)
    y:           The output of the forward_pass of NN for x  (np.array, must have the same shape as t)
    a:           The affine transforms from the forward_pass (np.array)
    z:           The activated units from the forward_pass (np.array)
    activations: The activations to be used                  (list of functions)
    loss:        The loss function to be used                (one function)
    
    Output:
    g_w: A list of gradients for every hidden unit 
    g_b: A list of gradients for every bias
    
    Shapes for the einsum:
    b: batch size
    i: size of the input hidden layer (layer l)
    o: size of the output (layer l+1)
    """
    BS = x.shape[0] # Implied batch shape 
    
    # First, let's compute the list of derivatives of z with respect to a 
    d_a = []
    for i in range(len(activations)):
        d_a.append(activations[i](a[i], derivative=True))
    
    # Second, let's compute the derivative of the loss function
    t = t.reshape(BS, -1)
    
    d_loss = loss(t,y,derivative=True) # <- Insert correct expression here
     
    # Third, let's compute the derivative of the biases and the weights
    g_w   = [] # List to save the gradient of the weights
    g_b   = [] # List to save the gradients of the biases

    # delta是d(loss)/d(a_1)，一共0,1,2三层，0是输入层，2是输出层，activation一个2个，0->1, 1->2，这个是1->2的
    delta = np.einsum('bo, bo -> bo', d_loss, d_a[-1])# loss shape: (b, o); pre-activation units shape: (b, o) hadamard product

    # 因为bias是作用在输出层的输出上的，所以不用管输出层的输入是什么尺寸，就是d(loss)/d(a_2)
    g_b.append(np.mean(delta, axis=0))
    # 输出层的输入和输出数分别是o和i
    g_w.append(np.mean(np.einsum('bo, bi -> bio', delta, z[-2]), axis=0)) # delta shape: (b, o), activations shape: (b, h) 

    for l in range(1, len(NN[0])):
        d_C_d_z = np.einsum('bo, io -> bi', delta, NN[0][-l])  # Derivative of the Cost with respect to an activated layer d_C_d_z. 
                                                               #  delta shape: as above; weights shape: (i, o)
                                                               # Delta: d_C_d_z (element-wise mult) derivative of the activation layers
                                                               #  delta shape: as above; d_z shape: (b, i)  
        delta = np.einsum('bo, bo->bo', d_C_d_z, d_a[-l-1])   # <- Insert correct expression 
                                                                

        g_b.append(np.mean(delta, axis=0)) 
        g_w.append(np.mean(np.einsum('bo, bi -> bio', delta, z[-l-2]), axis=0)) # Derivative of cost with respect to weights in layer l:
                                                                                # delta shape: as above; activations of l-1 shape: (b, i)
    
    return g_b[::-1], g_w[::-1]

def finite_diff_grad(x, NN, ACT_F, epsilon=None):
    """
    Finite differences gradient estimator: https://en.wikipedia.org/wiki/Finite_difference_method
    The idea is that we can approximate the derivative of any function (f) with respect to any argument (w) by evaluating the function at (w+e)
    where (e) is a small number and then computing the following opertion (f(w+e)-f(w))/e . Note that we would need N+1 evaluations of
    the function in order to compute the whole Jacobian (first derivatives matrix) where N is the number of arguments. The "+1" comes from the
    fact that we also need to evaluate the function at the current values of the argument.
    
    Input:
    x:       The point at which we want to evaluate the gradient
    NN:      The tuple that contains the neural network
    ACT_F:   The activation functions in order to perform the forward pass
    epsilon: The size of the difference
    
    Output:
    Two lists, the first one contains the gradients with respect to the weights, the second with respect to the biases
    """
    from copy import deepcopy
    
    if epsilon == None:
        epsilon = np.finfo(np.float32).eps # Machine epsilon for float 32
        
    grads = deepcopy(NN)               # Copy of structure of the weights and biases to save the gradients                        
    _ , test_z = forward_pass(x, NN_UT, ACT_F_UT) # We evaluate f(x)
    
    for e in range(len(NN)):                       # Iterator over elements of the NN:       weights or biases
        for h in range(len(NN[e])):                # Iterator over the layer of the element: layer number
            for r in range(NN[e][h].shape[0]):     # Iterator over                           row number
                for c in range(NN[e][h].shape[1]): # Iterator over                           column number 
                    NN_copy             = deepcopy(NN)    
                    NN_copy[e][h][r,c] += epsilon
                    _, test_z_eps       = forward_pass(x, NN_copy, ACT_F)     # We evaluate f(x+eps)
                    grads[e][h][r,c]    = (test_z_eps[-1]-test_z[-1])/epsilon # Definition of finite differences gradient
    
    return grads[0], grads[1]

# Initialize an arbitrary neural network
#L  = [3, 16, 1]
L  = [1, 8, 1]
NN = init_NN(L)
#NN = init_NN_glorot(L, uniform=True)
#NN = init_NN_he_ReLU(L, uniform=True)

ACT_F = [ReLU, Linear]
#ACT_F = [Tanh, Linear]

# Recommended hyper-parameters for 1-D: 
# L  = [1, 8, 1]
# EPOCHS = 10000
# BATCH_SIZE = 128 
# LEARN_R = 2.5e-1 for Tanh and LEARN_R = 1e-1 for ReLU

# Recommended hyper-parameters for 3-D: 
# L  = [3, 16, 1] 
# EPOCHS = 10000
# BATCH_SIZE = 128 
# LEARN_R = 5e-2 for ReLU and LEARN_R = 1e-1 for Tanh

### Notice that, when we switch from tanh to relu activation, we decrease the learning rate. This is due the stability of the gradients 
## of the activation functions.

# Initialize training hyperparameters
EPOCHS = 20000
BATCH_SIZE = 128 
LEARN_R = 1e-2 

train_loss = []
val_loss = []

for e in range(EPOCHS):
    # Mini-batch indexes
    idx = np.random.choice(x_train.shape[0], size=BATCH_SIZE)
    # Forward pass
    aff, units = forward_pass(x_train[idx,:], NN, ACT_F)
    # Backward pass
    g_b, g_w = backward_pass(x_train[idx,:], y_train[idx], units[-1], units, aff, NN, ACT_F, squared_error)
    
    # Stochastic gradient descent
    for l in range(len(g_b)):
        NN[0][l] -= LEARN_R*g_w[l]
        NN[1][l] -= LEARN_R*g_b[l]
        
    # Training loss
    _, units = forward_pass(x_train, NN, ACT_F)
    # Estimate loss function
    #print(np.max(squared_error(y_train, units[-1])))
    train_loss.append(np.mean(squared_error(y_train, np.squeeze(units[-1]))))
    
    # Validation
    # Forward pass
    _, units = forward_pass(x_validation, NN, ACT_F)
    # Estimate validation loss function
    val_loss.append(np.mean(squared_error(y_validation, np.squeeze(units[-1]))))
    
    if e%500==0:
        print("{:4d}".format(e),
              "({:5.2f}%)".format(e/EPOCHS*100), 
              "Train loss: {:4.3f} \t Validation loss: {:4.3f}".format(train_loss[-1], val_loss[-1]))
        
plt.plot(range(len(train_loss)), train_loss);
plt.plot(range(len(val_loss)), val_loss);
plt.show()