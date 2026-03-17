# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # A_prev: (batch_size, input_dim), W_curr: (output_dim, input_dim), b_curr: (output_dim, 1)
        # Z_curr: (batch_size, output_dim)
        Z_curr = A_prev @ W_curr.T + b_curr.T
        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        else:
            raise ValueError(f"Activation function {activation} not supported")
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """

        cache = {}
        A_curr = X
        cache['A0'] = X
        for i in range(len(self.arch)): # self.arch is a list of dictionaries
            layer_idx = i+1
            W_curr = self._param_dict[f'W{layer_idx}']
            b_curr = self._param_dict[f'b{layer_idx}']
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_curr, self.arch[i]['activation'])
            cache[f'Z{layer_idx}'] = Z_curr
            cache[f'A{layer_idx}'] = A_curr
        
        return A_curr, cache
        
        
    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # Apply activation backprop to get dZ from dA
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            raise ValueError(f"Activation function {activation_curr} not supported")

        # Compute gradients (row-convention: shapes are (batch_size, dim))
        m = A_prev.shape[0]
        dW_curr = dZ_curr.T @ A_prev / m
        db_curr = np.sum(dZ_curr, axis=0, keepdims=True).T / m
        dA_prev = dZ_curr @ W_curr

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {}
        n_layers = len(self.arch)

        # Compute initial gradient from loss function at the output layer
        if self._loss_func == 'binary_cross_entropy':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mean_squared_error':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise ValueError(f"Loss function {self._loss_func} not supported")

        # Backpropagate through each layer
        for idx in reversed(range(n_layers)):
            layer_idx = idx + 1

            W_curr = self._param_dict[f"W{layer_idx}"]
            b_curr = self._param_dict[f"b{layer_idx}"]
            A_prev = cache[f"A{idx}"]
            Z_curr = cache[f"Z{layer_idx}"]
            activation_curr = self.arch[idx]['activation']

            dA_prev, dW_curr, db_curr = self._single_backprop(
                W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr
            )
            grad_dict[f"dW{layer_idx}"] = dW_curr
            grad_dict[f"db{layer_idx}"] = db_curr
            dA_curr = dA_prev

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        # Update weights and biases using gradients and learning rate
        n_layers = len(self.arch)
        for idx in range(1, n_layers + 1):
            self._param_dict[f"W{idx}"] -= self._lr * grad_dict[f"dW{idx}"]
            self._param_dict[f"b{idx}"] -= self._lr * grad_dict[f"db{idx}"]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        per_epoch_loss_train = []
        per_epoch_loss_val = []

        epoch = 0
        while epoch < self._epochs:

            # Shuffle training data

            perm = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            # Create batches
            num_batches = int(np.ceil(X_train_shuffled.shape[0] / self._batch_size))
            X_batches = np.array_split(X_train_shuffled, num_batches)
            y_batches = np.array_split(y_train_shuffled, num_batches)

            batch_loss_train = []

            for X_batch, y_batch in zip(X_batches, y_batches):
                y_pred, cache = self.forward(X_batch)

                if self._loss_func == 'binary_cross_entropy':
                    train_loss = self._binary_cross_entropy(y_batch, y_pred)
                elif self._loss_func == 'mean_squared_error':
                    train_loss = self._mean_squared_error(y_batch, y_pred)
                else:
                    raise ValueError(f"Unsupported loss function: {self._loss_func}")

                batch_loss_train.append(train_loss)

                grad_dict = self.backprop(y_batch, y_pred, cache)

                # Update parameters in-place
                for idx in range(1, len(self.arch) + 1):
                    self._param_dict[f"W{idx}"] -= self._lr * grad_dict[f"dW{idx}"]
                    self._param_dict[f"b{idx}"] -= self._lr * grad_dict[f"db{idx}"]


            # Average batch losses to get per-epoch train loss
            per_epoch_loss_train.append(np.mean(batch_loss_train))

            # Validation loss for the epoch (on full validation set)
            y_val_pred = self.predict(X_val)
            if self._loss_func == 'binary_cross_entropy':
                val_loss = self._binary_cross_entropy(y_val, y_val_pred)
            elif self._loss_func == 'mean_squared_error':
                val_loss = self._mean_squared_error(y_val, y_val_pred)
            per_epoch_loss_val.append(val_loss)

            epoch += 1

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, cache = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        A = 1 / (1 + np.exp(-Z))
        return A

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        
        dZ = dA * self._sigmoid(Z) * (1 - self._sigmoid(Z))
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        A = np.maximum(0, Z)

        return A

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0 # for all non positive values the derivative is 0, for all other values it is itself
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # Epsilon to avoid log(0)
        eps = 1e-8  # Small constant to avoid log(0) and division by zero errors
        m = y.shape[1] if len(y.shape) > 1 else y.shape[0]  # Number of examples in the batch
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)  # Clip predictions so they are never exactly 0 or 1
        loss = -np.sum(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped)) / m  # Compute binary cross-entropy loss averaged over the batch
        return loss  # Return the computed average loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        eps = 1e-8  # Small constant to avoid division by zero
        m = y.shape[1] if len(y.shape) > 1 else y.shape[0]  # Number of examples in the batch
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        dA = -(y / y_hat_clipped) + (1 - y) / (1 - y_hat_clipped)
        dA = dA / m
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        m = y.shape[1] if len(y.shape) > 1 else y.shape[0]  # Proper handling for both 1D and 2D y
        loss = np.sum((y_hat - y) ** 2) / m  # Average loss over all elements
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        m = y.shape[1] if len(y.shape) > 1 else y.shape[0]  # Proper handling for both 1D and 2D y
        dA = 2 * (y_hat - y) / m  # Element-wise derivative, average over batch

        return dA