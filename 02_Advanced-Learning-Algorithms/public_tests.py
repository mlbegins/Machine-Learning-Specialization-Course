# UNIT TESTS

from tensorflow.keras.activations import sigmoid as tf_keras_sigmoid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, sigmoid, relu

import numpy as np
import tensorflow as tf

def compute_entropy_test(target):
    y = np.array([1] * 10)
    result = target(y)
    
    assert result == 0, "Entropy must be 0 with array of ones"
    
    y = np.array([0] * 10)
    result = target(y)
    
    assert result == 0, "Entropy must be 0 with array of zeros"
    
    y = np.array([0] * 12 + [1] * 12)
    result = target(y)
    
    assert result == 1, "Entropy must be 1 with same ammount of ones and zeros"
    
    y = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1])
    assert np.isclose(target(y), 0.918295, atol=1e-6), "Wrong value. Something between 0 and 1"
    assert np.isclose(target(-y + 1), target(y), atol=1e-6), "Wrong value"
    
    print("\033[92m All tests passed. ")
    
    
def split_dataset_test(target):

    # Case 1
    X = np.array([[1, 0], 
         [1, 0], 
         [1, 1], 
         [0, 0], 
         [0, 1]])
    X_t = np.array([[0, 1, 0, 1, 0]])
    X = np.concatenate((X, X_t.T), axis=1)

    left, right = target(X, list(range(5)), 2)
    expected = {'left': np.array([1, 3]),
                'right': np.array([0, 2, 4])}

    assert type(left) == list, f"Wrong type for left. Expected: list got: {type(left)}"
    assert type(right) == list, f"Wrong type for right. Expected: list got: {type(right)}"
    
    assert type(left[0]) == int, f"Wrong type for elements in the left list. Expected: int got: {type(left[0])}"
    assert type(right[0]) == int, f"Wrong type for elements in the right list. Expected: number got: {type(right[0])}"
    
    assert len(left) == 2, f"left must have 2 elements but got: {len(left)}"
    assert len(right) == 3, f"right must have 3 elements but got: {len(right)}"

    assert np.allclose(right, expected['right']), f"Wrong value for right. Expected: { expected['right']} \ngot: {right}"
    assert np.allclose(left, expected['left']), f"Wrong value for left. Expected: { expected['left']} \ngot: {left}"


    # Case 2
    X = np.array([[0, 1], 
         [1, 1], 
         [1, 1], 
         [0, 0], 
         [1, 0]])
    X_t = np.array([[0, 1, 0, 1, 0]])
    X = np.concatenate((X_t.T, X), axis=1)

    left, right = target(X, list(range(5)), 0)
    expected = {'left': np.array([1, 3]),
                'right': np.array([0, 2, 4])}


    assert len(left) == 2, f"left must have 2 elements but got: {len(left)}" 
    assert len(right) == 3, f"right must have 3 elements but got: {len(right)}"
    assert np.allclose(right, expected['right']) and np.allclose(left, expected['left']), f"Wrong value when target is at index 0."


    # Case 3
    X = (np.random.rand(11, 3) > 0.5) * 1 # Just random binary numbers
    X_t = np.array([[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]])
    X = np.concatenate((X, X_t.T), axis=1)

    left, right = target(X, [1, 2, 3, 6, 7, 9, 10], 3)
    expected = {'left': np.array([1, 3, 6]),
                'right': np.array([2, 7, 9, 10])}

    assert np.allclose(right, expected['right']) and np.allclose(left, expected['left']), f"Wrong value when target is at index 0. \nExpected: {expected} \ngot: \{left:{left}, 'right': {right}\}"
 
    
    print("\033[92m All tests passed.")
    
    
def compute_information_gain_test(target):
    X = np.array([[1, 0], 
         [1, 0], 
         [1, 0], 
         [0, 0], 
         [0, 1]])
    
    y = np.array([[0, 0, 0, 0, 0]]).T
    node_indexes = list(range(5))

    result1 = target(X, y, node_indexes, 0)
    result2 = target(X, y, node_indexes, 0)
    
    assert result1 == 0 and result2 == 0, f"Information gain must be 0 when target variable is pure. Got {result1} and {result2}"
    
    y = np.array([[0, 1, 0, 1, 0]]).T
    node_indexes = list(range(5))
    
    result = target(X, y, node_indexes, 0)
    assert np.isclose(result, 0.019973, atol=1e-6), f"Wrong information gain. Expected {0.019973} got: {result}"
    
    result = target(X, y, node_indexes, 1)
    assert np.isclose(result, 0.170951, atol=1e-6), f"Wrong information gain. Expected {0.170951} got: {result}"

    node_indexes = list(range(4))
    result = target(X, y, node_indexes, 0)
    assert np.isclose(result, 0.311278, atol=1e-6), f"Wrong information gain. Expected {0.311278} got: {result}"

    result = target(X, y, node_indexes, 1)
    assert np.isclose(result, 0, atol=1e-6), f"Wrong information gain. Expected {0.0} got: {result}"

    print("\033[92m All tests passed.")
    
    
def get_best_split_test(target):
    X = np.array([[1, 0], 
         [1, 0], 
         [1, 0], 
         [0, 0], 
         [0, 1]])

    y = np.array([[0, 0, 0, 0, 0]]).T
    node_indexes = list(range(5))

    result = target(X, y, node_indexes)
    
    assert result == -1, f"When the target variable is pure, there is no best split to do. Expected -1, got {result}"
    
    y = X[:,0]
    result = target(X, y, node_indexes)
    assert result == 0, f"If the target is fully correlated with other feature, that feature must be the best split. Expected 0, got {result}"
    y = X[:,1]
    result = target(X, y, node_indexes)
    assert result == 1, f"If the target is fully correlated with other feature, that feature must be the best split. Expected 1, got {result}"

    y = 1 - X[:,0]
    result = target(X, y, node_indexes)
    assert result == 0, f"If the target is fully correlated with other feature, that feature must be the best split. Expected 0, got {result}"

    y = np.array([[0, 1, 0, 1, 0]]).T
    result = target(X, y, node_indexes)
    assert result == 1, f"Wrong result. Expected 1, got {result}"

    y = np.array([[0, 1, 0, 1, 0]]).T    
    node_indexes = [2, 3, 4]
    result = target(X, y, node_indexes)
    assert result == 0, f"Wrong result. Expected 0, got {result}"

    n_samples = 100
    X0 = np.array([[1] * n_samples])
    X1 = np.array([[0] * n_samples])
    X2 = (np.random.rand(1, 100) > 0.5) * 1
    X3 = np.array([[1] * int(n_samples / 2) + [0] * int(n_samples / 2)])
    
    y = X2.T
    node_indexes = list(range(20, 80))
    X = np.array([X0, X1, X2, X3]).T.reshape(n_samples, 4)
    result = target(X, y, node_indexes)
    
    assert result == 2, f"Wrong result. Expected 2, got {result}"
    
    y = X0.T
    result = target(X, y, node_indexes)
    assert result == -1, f"When the target variable is pure, there is no best split to do. Expected -1, got {result}"
    print("\033[92m All tests passed.")
    
    
    
def test_my_softmax(target):
    z = np.array([1., 2., 3., 4.])
    a = target(z)
    atf = tf.nn.softmax(z)
    
    assert np.allclose(a, atf, atol=1e-10), f"Wrong values. Expected {atf}, got {a}"
    
    z = np.array([np.log(0.1)] * 10)
    a = target(z)
    atf = tf.nn.softmax(z)
    
    assert np.allclose(a, atf, atol=1e-10), f"Wrong values. Expected {atf}, got {a}"
    
    print("\033[92m All tests passed.")


def test_model(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    
    assert len(target.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, input_size], \
        f"Wrong input shape. Expected [None,  {input_size}] but got {target.input.shape.as_list()}"
    i = 0
    expected = [[Dense, [None, 25], relu],
                [Dense, [None, 15], relu],
                [Dense, [None, classes], linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        i = i + 1

    print("\033[92mAll tests passed!")
    

def test_c1(target):
    assert len(target.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, 400], \
        f"Wrong input shape. Expected [None,  400] but got {target.input.shape.as_list()}"
    i = 0
    expected = [[Dense, [None, 25], tf_keras_sigmoid],
                [Dense, [None, 15], tf_keras_sigmoid],
                [Dense, [None, 1], tf_keras_sigmoid]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        i = i + 1

    print("\033[92mAll tests passed!")
    
    
def test_c2(target):
    
    def linear(a):
        return a
    
    def linear_times3(a):
        return a * 3
    
    x_tst = np.array([1., 2., 3., 4.])  # (1 examples, 3 features)
    W_tst = np.array([[1., 2.], [1., 2.], [1., 2.], [1., 2.]]) # (3 input features, 2 output features)
    b_tst = np.array([0., 0.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape[0] == len(b_tst)
    assert np.allclose(A_tst, [10., 20.]), \
        "Wrong output. Check the dot product"
    
    b_tst = np.array([3., 5.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(A_tst, [13., 25.]), \
        "Wrong output. Check the bias term in the formula"
    
    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(A_tst, [39., 75.]), \
        "Wrong output. Did you apply the activation function at the end?"
    
    print("\033[92mAll tests passed!")  
    
    
def test_c3(target):
    
    def linear(a):
        return a
    
    def linear_times3(a):
        return a * 3
    
    x_tst = np.array([1., 2., 3., 4.])  # (1 examples, 3 features)
    W_tst = np.array([[1., 2.], [1., 2.], [1., 2.], [1., 2.]]) # (3 input features, 2 output features)
    b_tst = np.array([0., 0.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape[0] == len(b_tst)
    assert np.allclose(A_tst, [10., 20.]), \
        "Wrong output. Check the dot product"
    
    b_tst = np.array([3., 5.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(A_tst, [13., 25.]), \
        "Wrong output. Check the bias term in the formula"
    
    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(A_tst, [39., 75.]), \
        "Wrong output. Did you apply the activation function at the end?"
    
    x_tst = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])  # (2 examples, 4 features)
    W_tst = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12]]) # (3 input features, 2 output features)
    b_tst = np.array([0., 0., 0.])  # (2 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert A_tst.shape == (2, 3)
    assert np.allclose(A_tst, [[ 70.,  80.,  90.], [158., 184., 210.]]), \
        "Wrong output. Check the dot product"
    
    b_tst = np.array([3., 5., 6])  # (3 features)
    
    A_tst = target(x_tst, W_tst, b_tst, linear)
    assert np.allclose(A_tst, [[ 73.,  85.,  96.], [161., 189., 216.]]), \
        "Wrong output. Check the bias term in the formula"
    
    A_tst = target(x_tst, W_tst, b_tst, linear_times3)
    assert np.allclose(A_tst, [[ 219.,  255.,  288.], [483., 567., 648.]]), \
        "Wrong output. Did you apply the activation function at the end?"
    
    print("\033[92mAll tests passed!")  
