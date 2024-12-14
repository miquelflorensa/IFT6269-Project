# Tractable Gaussian Approximate Infernece
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


class TAGI:
    """
    Implementation of TAGI (Traceable Approximate Gaussian Inference) based on the paper:
    Tractable approximate Gaussian inference for Bayesian neural networks (James-A. Goulet, Luong-Ha Nguyen, and Said Amiri. JMLR, 2021)
    https://jmlr.csail.mit.edu/papers/volume22/20-1009/20-1009.pdf
    """

    def __init__(self, layers):
        """
        Initialize the TAGI model.

        Input:
        layers: list of integers, specifying the number of neurons in each layer.

        Output:
        Initializes the model parameters and hidden states.
        """
        self.layers = layers
        self.parameters = self.init_parameters(layers)
        self.hidden_states = self.init_hidden_states(layers)
        self.activated_states = self.init_hidden_states(layers)
        self.covariances = self.init_covariances(layers)
        self.mu = 0
        self.sigma = 1

    def standardize_data(self, X):
        """
        Standardize the input data by subtracting the mean and dividing by the standard deviation.

        Input:
        X: ndarray of shape (n_samples), the input data.

        Output:
        ndarray of shape (n_samples), the standardized data.
        """
        self.mu = np.mean(X)
        self.sigma = np.std(X)
        return (X - self.mu) / self.sigma

    def unstandardize_data(self, X):
        """
        Convert standardized data back to its original scale.

        Input:
        X: ndarray of shape (n_samples), the standardized data.

        Output:
        ndarray of shape (n_samples), the unstandardized data.
        """
        return X * self.sigma + self.mu

    def init_parameters(self, layers):
        """
        Initialize weights and biases for each layer.

        Input:
        layers: list of integers, specifying the number of neurons in each layer.

        Output:
        parameters: list of dictionaries, where each dictionary contains:
            - mw: ndarray of shape (n_inputs, n_outputs), the mean values of the weights.
            - Sw: ndarray of shape (n_inputs, n_outputs), the variances of the weights.
            - mb: ndarray of shape (n_outputs,), the mean values of the biases.
            - Sb: ndarray of shape (n_outputs,), the variances of the biases.
        """
        parameters = []
        for i in range(len(layers) - 1):
            input_size = layers[i]
            output_size = layers[i + 1]
            parameters.append(
                {
                    "mw": np.random.randn(output_size, input_size)
                    * (0.5 / input_size) ** 0.5,
                    "Sw": np.ones((output_size, input_size)) / input_size,
                    "mb": np.random.randn(output_size, 1) * 1 / input_size,
                    "Sb": np.ones((output_size, 1)) * 0.5,
                }
            )
        return parameters

    def init_hidden_states(self, layers):
        """
        Initialize the hidden states for each layer.

        Input:
        layers: list of integers, specifying the number of neurons in each layer.

        Output:
        hidden_states: list of dictionaries, where each dictionary contains:
            - 'ma': ndarray of shape (n_neurons,1), the mean values of the activation units.
            - 'Sa': ndarray of shape (n_neurons,n_neurons), the variances of the activation units.
            - 'J': ndarray of shape (n_neurons,1), the Jacobian of the activation function.
        """
        hidden_states = []
        for layer_size in layers:
            hidden_states.append(
                {
                    "ma": np.zeros((layer_size, 1)),
                    "Sa": np.ones((layer_size, 1)),
                    "J": np.ones((layer_size, 1)),
                }
            )
        return hidden_states

    def init_covariances(self, layers):
        """
        Initialize the covariances for each layer.

        Input:
        layers: list of integers, specifying the number of neurons in each layer.

        Output:
        covariances: list of dictionaries, where each dictionary contains:
            - 'covZw': ndarray of shape (n_inputs, n_outputs), the covariances of the hidden states and weights.
            - 'covZb': ndarray of shape (n_outputs,), the covariances of the hidden states and biases.
            - 'covZZ': ndarray of shape (n_neurons, n_neurons), the covariances of two consecutive hidden states.
        """
        covariances = []
        for i in range(len(layers) - 1):
            input_size = layers[i]
            output_size = layers[i + 1]
            covariances.append(
                {
                    "covZw": np.zeros((output_size, input_size)),
                    "covZb": np.zeros((output_size, 1)),
                    "covZZ": np.zeros((output_size, input_size)),
                }
            )

        return covariances

    def GMA(self, ma, Sa, mw, Sw):
        """
        Gaussian Multiplicative Approximation (GMA)

        Input:
        ma: ndarray of shape (n_inputs,), mean values of the activation units.
        Sa: ndarray of shape (n_inputs,), variances of the activation units.
        mw: ndarray of shape (n_inputs, n_outputs), mean values of the weights.
        Sw: ndarray of shape (n_inputs, n_outputs), variances of the weights.

        Output:
        mu: ndarray of shape (n_outputs,), the means of the next hidden layer.
        var: ndarray of shape (n_outputs,), the variances of the next hidden layer.

        Remember that in GMA, given X1 and X2 random gaussian variables,
        Expected value: E[X1 * X2] = μ1 * μ2 + cov(X1, X2)
        Variance: var[X1 * X2] = var(X1) * var(X2) + cov(X1, X2)^2 +
                  2cov(X1,X2)μ1μ2 + var(X1) * μ2^2 + var(X2) * μ1^2

        Also remebmer that the activation units and the weights are independent in a neural network.
        """
        # To-do: Implement GMA and compute mu and var.

        # Expected value E[X1 * X2] = μ1 * μ2
        mu = mw @ ma  # Matrix product of means

        # Variance calculation
        # var[X1 * X2] = Sa * Sw + (Sa * mw^2) + (Sw * ma^2)
        var = (Sw @ Sa) + (Sw @ ma**2) + (mw**2 @ Sa)

        return mu, var

    def ReLU(self, mz, Sz):
        """
        Rectified Linear Unit (ReLU) activation function.

        Input:
        mz: ndarray of shape (n_samples,), the hidden states.
        Sz: ndarray of shape (n_samples,), the variances of the hidden states.

        Output:
        ndarray of shape (n_samples,), the output values.
        """

        # To-do: Implement ReLU activation function and compute ma, Sa and J
        ma = np.maximum(0, mz)
        Sa = Sz * (mz > 0)
        J = (mz > 0).astype(int)

        return ma, Sa, J

    def forward(self, X):
        print("forward")
        """
        Perform a forward pass through the network using GMA.

        Input:
        X: ndarray of shape (n_inputs,), the input data.

        Output:
        Y: ndarray of shape (n_outputs,), the output values.

        It is suggested to compute the coariances at the same time you perform the forward pass.
        """

        # To-do: Implement forward pass using GMA and compute the covariances.
        self.hidden_states[0]["ma"], self.hidden_states[0]["Sa"] = X, np.zeros_like(X)
        self.activated_states[0]["ma"], self.activated_states[0]["Sa"] = (
            X,
            np.zeros_like(X),
        )

        for i in range(len(self.layers) - 1):
            ma, Sa = self.activated_states[i]["ma"], self.activated_states[i]["Sa"]
            mw, Sw = self.parameters[i]["mw"], self.parameters[i]["Sw"]
            mb, Sb = self.parameters[i]["mb"], self.parameters[i]["Sb"]

            # Apply GMA
            mz, Sz = self.GMA(ma, Sa, mw, Sw)
            mz = mz + mb
            Sz = Sz + Sb

            # Apply ReLU if not the output layer
            J = np.ones_like(mz)
            if i < len(self.layers) - 2:
                ma, Sa, J = self.ReLU(mz, Sz)
            else:
                ma, Sa = mz, Sz

            self.hidden_states[i + 1] = {"ma": mz, "Sa": Sz, "J": J}
            self.activated_states[i + 1] = {"ma": ma, "Sa": Sa, "J": J}

            self.covariances[i]["covZw"] = Sw * mz 
            self.covariances[i]["covZb"] = Sb
            # self.covariances[i]["covZZ"] = (Sz @ J.T @ mw).T  # TODO: Check this
            self.covariances[i]["covZZ"] = Sz * J * mw
            print(self.covariances[i]["covZZ"])
            print(self.covariances[i]["covZZ"].shape)
            print(self.covariances[i]["covZw"])
            print(self.covariances[i]["covZw"].shape)
            print(self.covariances[i]["covZb"])
            print(self.covariances[i]["covZb"].shape)
            break
            # self.covariances[i]["covZZ"] = mw * Sz * J
            # clip covzz
            # self.covariances[i]["covZZ"] = np.clip(self.covariances[i]["covZZ"], -1, 1)


        # Output layer
        return self.activated_states[-1]["ma"], self.activated_states[-1]["Sa"]

    def backward(self, Y, sigma_v):
        print("backward")
        """
        Perform a backward pass through the network using Bayesian inference.

        Input:
        Y: ndarray of shape (n_outputs,), the output values.
        sigma_v: float, the observation noise.

        We are just gonna update the parameters and not the hidden states.

        Infer posterior mean and diagonal covariance for the output layer:
        - f(z_out|y) = N(z_out; mu_{z_out|y}, Sigma_{z_out|y})
        - mu_{z_out|y} = mu_{z_out} + cov_{Y, Z_out}^T * cov_{Y}^{-1} * (y - mu_Y)
        - Sigma_{z_out|y} = Sigma_{z_out} - cov_{Y, Z_out}^T * cov_{Y}^{-1} * cov_{Y, Z_out}

        RTS smoother for the parameters:
        - f(\theta|y) = N(\theta; mu_{\theta|y}, Sigma_{\theta|y})
        - mu_{\theta|y} = mu_{\theta} + J_{\theta} * (mu_{Z+|y} - mu_{Z+})
        - Sigma_{\theta|y} = Sigma_{\theta} + J_{\theta} * (Sigma_{Z+|y} - Sigma_{Z+}) * J_{\theta}^T

            where:
            - J_{\theta} = cov_{\theta, Z+} * cov_{Z+}^{-1}
        """
        # store the forward pass hidden states
        forward_mz = [hs["ma"].copy() for hs in self.hidden_states]
        forward_Sz = [hs["Sa"].copy() for hs in self.hidden_states]

        # Update output layer
        mu_y = self.hidden_states[-1]["ma"]  # Predicted mean of the output
        Sigma_y = (
            self.hidden_states[-1]["Sa"] + sigma_v**2
        )  # Total variance of the output (add noise)
        cov_yZ = (
            self.hidden_states[-1]["Sa"] / Sigma_y
        )  # Covariance between output and last hidden state

        # Infer posterior for the output layer
        residual = Y - mu_y  # Error between predicted and true output
        mu_z_out_y = mu_y + cov_yZ * residual  # Posterior mean of the last hidden state
        Sigma_z_out_y = (
            self.hidden_states[-1]["Sa"] - cov_yZ**2 * Sigma_y
        )  # Posterior variance of the last hidden state

        # Update the output layer's hidden state parameters
        self.hidden_states[-1]["ma"] = mu_z_out_y
        self.hidden_states[-1]["Sa"] = Sigma_z_out_y

        # RTS smoother for parameters
        for i in reversed(range(len(self.layers) - 1)):
            # Calculate the Jacobian
            # denominator = forward_Sz[i + 1]
            denominator = self.hidden_states[i+1]["Sa"]
            J_z = np.where(
                denominator != 0, self.covariances[i]["covZZ"]/denominator, 0
            )
            J_w = np.where(
                denominator != 0, self.covariances[i]["covZw"] / denominator, 0
            )
            J_b = np.where(
                denominator != 0, self.covariances[i]["covZb"] / denominator, 0
            )

            # Update posterior for the current layer's parameters
            delta_mu_z = self.hidden_states[i + 1]["ma"] - forward_mz[i + 1]
            delta_Sigma_z = self.hidden_states[i + 1]["Sa"] - forward_Sz[i + 1]

            # Compute the updated mean and variance of the hidden states
            mu_z_y = self.hidden_states[i]["ma"] + J_z * delta_mu_z
            Sigma_z_y = self.hidden_states[i]["Sa"] + delta_Sigma_z * J_z**2

            # Compute updated mean and variance for parameters
            mu_w_y = self.parameters[i]["mw"] + J_w * delta_mu_z
            Sigma_w_y = self.parameters[i]["Sw"] + delta_Sigma_z * (J_w**2)
            mu_b_y = self.parameters[i]["mb"] + J_b * delta_mu_z
            Sigma_b_y = self.parameters[i]["Sb"] + delta_Sigma_z * (J_b**2)

            # Save updated parameter posterior
            self.parameters[i]["mw"] = mu_w_y
            self.parameters[i]["Sw"] = Sigma_w_y

            self.parameters[i]["mb"] = mu_b_y
            self.parameters[i]["Sb"] = Sigma_b_y

            # Save updated hidden state posterior
            self.hidden_states[i]["ma"] = mu_z_y
            self.hidden_states[i]["Sa"] = Sigma_z_y


# Generate data
X = np.linspace(0, 10, 50).reshape(-1, 1)  # Ensure X is 2D
# Y = np.sin(X).flatten() + np.random.normal(0, 0.1, 50) + np.random.normal(0, 0.05, 50)
Y = X.flatten() + np.random.normal(0, 0.5, 50)

# Shuffle data
indices = np.arange(50)
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices].reshape(-1, 1)

# Split data
train_size = int(0.8 * 50)
train_X, test_X = X[:train_size], X[train_size:]
train_Y, test_Y = Y[:train_size], Y[train_size:]

# Plot the training data
plt.scatter(train_X, train_Y, marker="o", color="b")

# Train
net = TAGI([1, 50, 1])  # define model
train_X = net.standardize_data(train_X)
train_Y = net.standardize_data(train_Y)


for i in range(len(net.layers) - 1):
    print(
        net.parameters[i]["mw"].shape,
        net.parameters[i]["Sw"].shape,
        net.parameters[i]["mb"].shape,
        net.parameters[i]["Sb"].shape,
    )
    print(
        net.hidden_states[i]["ma"].shape,
        net.hidden_states[i]["Sa"].shape,
        net.hidden_states[i]["J"].shape,
    )
    print(
        net.covariances[i]["covZw"].shape,
        net.covariances[i]["covZb"].shape,
        net.covariances[i]["covZZ"].shape,
    )
    print("-------------------")

# net.forward(train_X[0])
# net.backward(train_Y[1], 0.01)

epochs = 1
for epoch in range(epochs):
    for x, y in zip(train_X, train_Y):
        x = x.reshape(-1, 1)
        net.forward(x)
        net.backward(y, 0.01)

# Predict
test_Y_pred = []
test_S_pred = []
for x in test_X:
    x = x.reshape(-1, 1)
    m_pred, S_pred = net.forward(x)
    test_Y_pred.append(m_pred[0])
    test_S_pred.append(S_pred[0][0])

test_Y_pred = np.array(test_Y_pred).flatten()
test_S_pred = np.array(test_S_pred).flatten()

test_Y_pred = net.unstandardize_data(test_Y_pred)
test_S_pred = test_S_pred * net.sigma**2  # Unstandardize variance
test_S_pred = np.sqrt(test_S_pred)  # Convert variance to standard deviation

# Plot
plt.scatter(test_X.flatten(), test_Y, color="blue", label="True")
plt.scatter(test_X.flatten(), test_Y_pred, color="red", label="Predicted")
plt.fill_between(
    test_X.flatten(),
    test_Y_pred - test_S_pred,
    test_Y_pred + test_S_pred,
    color="red",
    alpha=0.2,
    label="Uncertainty",
)
plt.legend()
plt.show()
