import numpy as np
import random


def test_cofi_cost_func(target):
    num_users_r = 4
    num_movies_r = 5 
    num_features_r = 3

    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.zeros((1, num_users_r))
    Y_r = np.zeros((num_movies_r, num_users_r))
    R_r = np.zeros((num_movies_r, num_users_r))
    
    J = target(X_r, W_r, b_r, Y_r, R_r, 2);
    assert not np.isclose(J, 13.5), f"Wrong value. Got {J}. Did you multiply the regularization term by lambda_?"
    assert np.isclose(J, 27), f"Wrong value. Expected {27}, got {J}. Check the regularization term"
    
    
    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.ones((1, num_users_r))
    Y_r = np.ones((num_movies_r, num_users_r))
    R_r = np.ones((num_movies_r, num_users_r))

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 0);
    
    assert np.isclose(J, 90), f"Wrong value. Expected {90}, got {J}. Check the term without the regularization"
    
    
    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.ones((1, num_users_r))
    Y_r = np.zeros((num_movies_r, num_users_r))
    R_r = np.ones((num_movies_r, num_users_r))

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 0);
    
    assert np.isclose(J, 160), f"Wrong value. Expected {160}, got {J}. Check the term without the regularization"
    
    X_r = np.ones((num_movies_r, num_features_r))
    W_r = np.ones((num_users_r, num_features_r))
    b_r = np.ones((1, num_users_r))
    Y_r = np.ones((num_movies_r, num_users_r))
    R_r = np.ones((num_movies_r, num_users_r))

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 1);
    
    assert np.isclose(J, 103.5), f"Wrong value. Expected {103.5}, got {J}. Check the term without the regularization"
    
    num_users_r = 3
    num_movies_r = 4 
    num_features_r = 4
    
    #np.random.seed(247)
    X_r = np.array([[0.36618032, 0.9075415,  0.8310605,  0.08590986],
                     [0.62634721, 0.38234325, 0.85624346, 0.55183039],
                     [0.77458727, 0.35704147, 0.31003294, 0.20100006],
                     [0.34420469, 0.46103436, 0.88638208, 0.36175401]])#np.random.rand(num_movies_r, num_features_r)
    W_r = np.array([[0.04786854, 0.61504665, 0.06633146, 0.38298908], 
                    [0.16515965, 0.22320207, 0.89826005, 0.14373251], 
                    [0.1274051 , 0.22757303, 0.96865613, 0.70741111]])#np.random.rand(num_users_r, num_features_r)
    b_r = np.array([[0.14246472, 0.30110933, 0.56141144]])#np.random.rand(1, num_users_r)
    Y_r = np.array([[0.20651685, 0.60767914, 0.86344527], 
                    [0.82665019, 0.00944765, 0.4376798 ], 
                    [0.81623732, 0.26776794, 0.03757507], 
                    [0.37232161, 0.19890823, 0.13026598]])#np.random.rand(num_movies_r, num_users_r)
    R_r = np.array([[1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]])#(np.random.rand(num_movies_r, num_users_r) > 0.4) * 1

    # Evaluate cost function
    J = target(X_r, W_r, b_r, Y_r, R_r, 3);
    
    assert np.isclose(J, 13.621929978531858, atol=1e-8), f"Wrong value. Expected {13.621929978531858}, got {J}."
    
    print('\033[92mAll tests passed!')
    

def select_threshold_test(target):
    p_val = np.array([i / 100 for i in range(30)])
    y_val = np.array([1] * 5 + [0] * 25)
    
    best_epsilon, best_F1 = target(y_val, p_val)
    assert np.isclose(best_epsilon, 0.04, atol=0.3 / 1000), f"Wrong best_epsilon. Expected: {0.04} got: {best_epsilon}"
    assert best_F1 == 1, f"Wrong best_F1. Expected: 1 got: {best_F1}"
    
    y_val = np.array([1] * 5 + [0] * 25)
    y_val[2] = 0 # Introduce noise
    best_epsilon, best_F1 = target(y_val, p_val)
    assert np.isclose(best_epsilon, 0.04, atol=0.3 / 1000), f"Wrong best_epsilon. Expected: {0.04} got: {best_epsilon}"
    assert np.isclose(best_F1, 0.8888888), f"Wrong best_F1. Expected: 0.8888888 got: {best_F1}"
    
    p_val = np.array([i / 1000 for i in range(50)])
    y_val = np.array([1] * 8 + [0] * 42)
    y_val[5] = 0
    index = [*range(50)]
    random.shuffle(index)
    p_val = p_val[index]
    y_val = y_val[index]

    best_epsilon, best_F1 = target(y_val, p_val)
    assert np.isclose(best_epsilon, 0.007, atol=0.05 / 1000), f"Wrong best_epsilon. Expected: {0.0070070} got: {best_epsilon}"
    assert np.isclose(best_F1, 0.933333333), f"Wrong best_F1. Expected: 0.933333333 got: {best_F1}"
    print("\033[92mAll tests passed!")
    
    
def estimate_gaussian_test(target):
    np.random.seed(273)
    
    X = np.array([[1, 1, 1], 
                  [2, 2, 2], 
                  [3, 3, 3]]).T
    
    mu, var = target(X)
    
    assert type(mu) == np.ndarray, f"Wrong type for mu. Expected: {np.ndarray} got: {type(mu)}"
    assert type(var) == np.ndarray, f"Wrong type for var. Expected: {np.ndarray} got: {type(var)}"
    
    assert mu.shape == (X.shape[1],), f"Wrong shape for mu. Expected: {(X.shape[1],)} got: {mu.shape}"
    assert type(var) == np.ndarray, f"Wrong shape for var. Expected: {(X.shape[1],)} got: {var.shape}"
    
    assert np.allclose(mu, [1., 2., 3.]), f"Wrong value for mu. Expected: {[1, 2, 3]} got: {mu}"
    assert np.allclose(var, [0., 0., 0.]), f"Wrong value for var. Expected: {[0, 0, 0]} got: {var}"
    
    X = np.array([[1, 2, 3], 
                  [2, 4, 6], 
                  [3, 6, 9]]).T
    
    mu, var = target(X)
    
    assert type(mu) == np.ndarray, f"Wrong type for mu. Expected: {np.ndarray} got: {type(mu)}"
    assert type(var) == np.ndarray, f"Wrong type for var. Expected: {np.ndarray} got: {type(var)}"
    
    assert mu.shape == (X.shape[1],), f"Wrong shape for mu. Expected: {(X.shape[1],)} got: {mu.shape}"
    assert type(var) == np.ndarray, f"Wrong shape for var. Expected: {(X.shape[1],)} got: {var.shape}"
    
    assert np.allclose(mu, [2., 4., 6.]), f"Wrong value for mu. Expected: {[2., 4., 6.]} got: {mu}"
    assert np.allclose(var, [2. / 3, 8. / 3., 18. / 3.]), f"Wrong value for var. Expected: {[2. / 3, 8. / 3., 18. / 3.]} got: {var}"
    
    
    m = 500
    X = np.array([np.random.normal(0, 1, m), 
                  np.random.normal(1, 2, m), 
                  np.random.normal(3, 1.5, m)]).T
    
    mu, var = target(X)
    
    assert type(mu) == np.ndarray, f"Wrong type for mu. Expected: {np.ndarray} got: {type(mu)}"
    assert type(var) == np.ndarray, f"Wrong type for var. Expected: {np.ndarray} got: {type(var)}"
    
    assert mu.shape == (X.shape[1],), f"Wrong shape for mu. Expected: {(X.shape[1],)} got: {mu.shape}"
    assert type(var) == np.ndarray, f"Wrong shape for var. Expected: {(X.shape[1],)} got: {var.shape}"
    
    assert np.allclose(mu, [0., 1., 3.], atol=0.2), f"Wrong value for mu. Expected: {[0, 1, 3]} got: {mu}"
    assert np.allclose(var, np.square([1., 2., 1.5]), atol=0.2), f"Wrong value for var. Expected: {np.square([1., 2., 1.5])} got: {var}"
    
    print("\033[92mAll tests passed!")

def compute_centroids_test(target):
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [-1.1, -1.7], [-1.6, 1.2]])
    idx = np.array([1, 1, 1, 0, 0, 0, 2])
    K = 3
    centroids = target(X, idx, K)
    expected_centroids = np.array([[0.13333333,  0.43333333],
                                   [-1.33333333, -0.5      ],
                                   [-1.6,        1.2       ]])
    
    assert type(centroids) == np.ndarray, "Wrong type"
    assert centroids.shape == (K, X.shape[1]), f"Wrong shape. Expected: {centroids.shape} got: {(K, X.shape[1])}"
    assert np.allclose(centroids, expected_centroids), f"Wrong values. Expected: {expected_centroids}, got: {centroids}"
    
    X = np.array([[2, 2.5], [2.5, 2.5], [-1.5, -1.5],
                  [2, 2], [-1.5, -1], [-1, -1]])
    idx = np.array([0, 0, 1, 0, 1, 1])
    K = 2
    centroids = target(X, idx, K)
    expected_centroids = np.array([[[ 2.16666667,  2.33333333],
                                    [-1.33333333, -1.16666667]]])
    
    assert type(centroids) == np.ndarray, "Wrong type"
    assert centroids.shape == (K, X.shape[1]), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(centroids, expected_centroids), f"Wrong values. Expected: {expected_centroids}, got: {centroids}"
    
    print("\033[92mAll tests passed!")
    
def find_closest_centroids_test(target):
    # With 2 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, -1],
                  [2, 2],[2.5, 2.5],[2, 2.5]])
    initial_centroids = np.array([[-1, -1], [2, 2]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [0, 0, 0, 1, 1, 1]), "Wrong values"
    
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [2, 2]])
    initial_centroids = np.array([[2.5, 2], [-1, -1], [-1.5, 1.]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [1, 1, 2, 2, 0, 0]), f"Wrong values. Expected {[2, 2, 0, 0, 1, 1]}, got: {idx}"
    
    # With 3 centroids
    X = np.array([[-1, -1], [-1.5, -1.5], [-1.5, 1],
                  [-1, 1.5], [2.5, 1.5], [-1.1, -1.7], [-1.6, 1.2]])
    initial_centroids = np.array([[2.5, 2], [-1, -1], [-1.5, 1.]])
    idx = target(X, initial_centroids)
    
    assert type(idx) == np.ndarray, "Wrong type"
    assert idx.shape == (len(X),), f"Wrong shape. Expected: {(len(X),)} got: {idx.shape}"
    assert np.allclose(idx, [1, 1, 2, 2, 0, 1, 2]), f"Wrong values. Expected {[2, 2, 0, 0, 1, 1]}, got: {idx}"
    
    print("\033[92mAll tests passed!")