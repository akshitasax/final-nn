import numpy as np
import pytest

from nn.preprocess import sample_seqs, one_hot_encode_seqs
from nn.nn import NeuralNetwork

def test_single_forward():
    # Single data point, simple network (input_dim=2, output_dim=1)
    nn_arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    nn = NeuralNetwork(nn_arch, lr=0.1, seed=1, batch_size=1, epochs=1, loss_function="binary_cross_entropy")
    X = np.array([[0.5, -0.2]])
    # Should return a prediction between 0 and 1
    out = nn._single_forward(X)
    assert out.shape == (1, 1)
    assert 0 <= out[0, 0] <= 1

def test_forward():
    # Test on batch, input_dim=2, output_dim=1, batch size 3
    nn_arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    nn = NeuralNetwork(nn_arch, lr=0.1, seed=1, batch_size=3, epochs=1, loss_function="binary_cross_entropy")
    X = np.array([
        [0.1, 0.2],
        [0.2, 0.3],
        [-0.1, 0.5]
    ])
    out = nn._forward(X)
    assert out.shape == (3, 1)
    assert np.all((out >= 0) & (out <= 1))

def test_single_backprop():
    # Create simple network and fake forward pass output and label for backprop
    nn_arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    nn = NeuralNetwork(nn_arch, lr=0.1, seed=0, batch_size=1, epochs=1, loss_function="binary_cross_entropy")
    X = np.array([[0.1, 0.2]])
    y = np.array([[1]])
    output = nn._single_forward(X)
    grads = nn._single_backprop(X, y)
    # Should return dict of gradients for each parameter
    assert isinstance(grads, dict)
    # Check expected keys
    for l in range(1, 2):
        assert f"W{l}" in grads and f"b{l}" in grads

def test_predict():
    nn_arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    nn = NeuralNetwork(nn_arch, lr=0.1, seed=1, batch_size=1, epochs=1, loss_function="binary_cross_entropy")
    # Two samples, one likely >0.5, one <0.5
    X = np.array([[10, 10], [-10, -10]])
    preds = nn.predict(X)
    assert preds.shape == (2,)
    assert np.all((preds == 0) | (preds == 1))

def test_binary_cross_entropy():
    # Test typical BCE usage
    from nn.nn import binary_cross_entropy
    y_true = np.array([[1], [0], [1], [0]])
    y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])
    loss = binary_cross_entropy(y_true, y_pred)
    assert np.isscalar(loss) or loss.shape == ()
    # Perfect match case: loss near zero
    p2 = np.array([[1], [0], [1], [0]])
    assert binary_cross_entropy(y_true, p2) < 1e-4

def test_binary_cross_entropy_backprop():
    from nn.nn import binary_cross_entropy_backprop
    y_true = np.array([[1], [0]])
    y_pred = np.array([[0.9], [0.1]])
    grad = binary_cross_entropy_backprop(y_true, y_pred)
    # Should have same shape
    assert grad.shape == y_true.shape
    # Check sign: dL/dy_pred is negative for y_true=1 and y_pred<1
    assert grad[0, 0] < 0
    assert grad[1, 0] > 0

def test_mean_squared_error():
    from nn.nn import mean_squared_error
    y_true = np.array([[1], [0], [1]])
    y_pred = np.array([[1], [0], [1]])
    loss = mean_squared_error(y_true, y_pred)
    assert loss == 0.0
    y_pred2 = np.array([[0.5], [0.5], [0.5]])
    loss2 = mean_squared_error(y_true, y_pred2)
    assert loss2 > 0

def test_mean_squared_error_backprop():
    from nn.nn import mean_squared_error_backprop
    y_true = np.array([[1.0], [0.0]])
    y_pred = np.array([[0.7], [0.3]])
    grad = mean_squared_error_backprop(y_true, y_pred)
    assert grad.shape == y_true.shape
    # Check math: grad = 2*(y_pred - y_true)/n
    n = len(y_true)
    expected = 2 * (y_pred - y_true) / n
    assert np.allclose(grad, expected)

def test_sample_seqs():
    seqs = ["ACGT", "TGCA", "CCCC", "GGGG", "TTTT"]
    labels = [True, False, False, True, False]
    np.random.seed(0)
    import random; random.seed(0)
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)
    # Should be balanced
    pos = sum(sampled_labels)
    neg = len(sampled_labels) - pos
    assert pos == neg

def test_one_hot_encode_seqs():
    # Test canonical bases
    seqs = ["A", "T", "C", "G", "AGT"]
    encoded = one_hot_encode_seqs(seqs)
    # Each A/T/C/G becomes [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
    expected = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [1,0,0,0, 0,0,0,1, 0,1,0,0]
    ]
    for i in range(len(seqs)):
        assert np.array_equal(encoded[i], expected[i])
    # Test unknown base encoded as [0,0,0,0]
    seqs2 = ["N"]
    enc2 = one_hot_encode_seqs(seqs2)
    assert np.all(enc2[0] == [0,0,0,0])