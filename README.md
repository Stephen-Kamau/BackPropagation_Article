Backpropagation: Understanding how Backpropagation Algorithm Works in ML
Introduction.
Backpropagation is one of the common term while one is learning Introduction to Neural Networks. It is an algorithm used in the training of neural networks to adjust the weights of a single neurons by moving backward from the output of the neuron. It involves the process of adjusting the weights of the inputs to an neural network in order to minimize the errors. The process starts with randomly generating weights for the network, then using backpropagation to optimize them to the model. This is one of the essential component of a Neural Network for it to have good perfomances.

What Does Backpropagation Algorithm involves?
It is a supervised learning algorithm used mostly for optimization of weights and biases while Training a Neural Network. It is extracted from chain rules in calculus where it is used to find gradient of the loss function against the weights if the neural network(NN). It works by moving or propagating errror from the output layer backward into the NN, each layer by layer while adjsuting their weights based on the gradient achieved.

The gradient of the loss function for each weights is used to update the weight in the opposite direction of the gradient (Backward), which is the direction that minimizes the loss function. This process is repeated until the loss function is minimized upto a certain set threshold or the number of iterations is reached.

The working of Backpropagation Algorithm?
It works through minimization of error by adjusting the weights in the network between the predicted output and the actual output. At start, random weights are initialized for each neuron in the network. The input is the fed to the network and the output is calculated by using the network's weights against the inputs. Since the algorithm is supervised, the actual output is used against the predicted output to guide the algorithm what intensity should weights be updated.

This is done by getting the error between the predictions and the actual output. The error is the propagated back through the network layer by layer for each neuron by the use of chain rule. The error is used to get the gradient function that is used to adjust the weights in each layer and the process is done repeatedly until the conditions required are met.



Maths of Backpropagation Algorithm.
For this algorithm, there are normally two parts i.e the forward pass and backward pass

1. Forward Pass
This is the process of moving input data through the network in order to generated output. It moves inputs in forward manner. A product of weights and input data is done in the first layer then passed through an activation function of choice and biases added. The output from the first layer is then multiplied by weights in second layer with biases addition and then also passed through the activation fucntion selected. This process is continued upto the output layer. The output of each layer can be represented mathematically as follows:

Z = (WX + B)

where Z is the output of the ith layer, W is the weight matrix for the ith layer, X is the input to the network, and B is the bias vector for the ith layer.

The output of each layer is passed through the activation function to generate the output of that layer. The activation function is typically a nonlinear function that adds nonlinearity to the network to increase the perfomance. The output of each layer can be represented mathematically as follows:

Y = f(Z)

where Y is the output of the ith layer after passing through the activation function f.

2. Backward Pass
In this part, the process  involves propagating the error back through the network to adjust the weights. The error is calculated as the difference between the predicted output and the actual output. The error is then propagated back through the network using the chain rule of calculus to calculate the derivative of the loss function with respect to each weight. Here is what happens in backward pass;
  i. Initialize the gradient of the loss function with respect to the output of the network (dL/dy_hat) to be the gradient of the loss function with respect to the predicted output (y_hat) of the last layer.
  ii. Compute the gradient of the loss function with respect to the parameters of the last layer (dL/dw2 and dL/db2) using the chain rule.
  iii. Propagate the gradient back to the previous layer by computing the gradient of the loss function with respect to the output of the previous layer (dL/dh1) using the chain rule.
  iv. Compute the gradient of the loss function with respect to the parameters of the previous layer (dL/dw1 and dL/db1) using the chain rule.
  v. Repeat steps 3-4 until the gradient of the loss function with respect to the input (dL/dx) has been computed.
  vi. Update the parameters (weights and biases) of the network using an optimization algorithm, such as stochastic gradient descent, using the computed gradients to minimize the loss function.


Here is the formula that can be used for backward pass;
```
dY = Y - Y_out  # Compute the error between the predicted output and the true output
dW2 = LR * dY * Sig(Y_out) * sigD(Y_out)  # Compute the derivative of the error with respect to the weights of the output layer
dW1 = LR * dW2 * Sig(hidden_layer) * sigD(hidden_layer) * W2.T  # Compute the derivative of the error with respect to the weights of the hidden layer

Where Y is expected output, Y_hat is predicted output, Sig is sigmoid activation function, sigD is derivative of sigmoid function, LR is learning rate(determines step size of weights update), W is the weights and dW is the gradient of the loss function with respect to the weights, computed using the backpropagation algorithm.


```

Once we are able to find derivative of the error, the weights are then updated as follows;
```
W = W - LR * dW

```
Once we have updated the weights for all layers, we go back to the forward pass and calculate the output again using the updated weights. We then compare the new output to the expected output and calculate the error. We continue this process of forward and backward passes until the error is minimized.


The training Process with an Example.
In our example, we will be doing a simulation of a neural Network that uses Backpropagation to train the model. This is a simple test just for the purpose of showcasing the working. The code is well commented for each process and can be easily Understood. It start by moving the input through the network using a forward pass method while calculating the error between the predicted output and the expected output using the mean squared error function. We then calculate the delta values for each layer using the derivative of the sigmoid function. We use these delta values to update the weights for each layer using the learning rate. The code snippet below shows this;

```python
#get the input data
l1 = X
#get the output from layer two
l2 = self.sigmoid_ftn(np.dot(l1, self.w1))
#output from layer 3 which is prediction
l3 = self.sigmoid_ftn(np.dot(l2, self.w2))

#get the error which is difference between training preds and actual value
target_pred_errors = Y - l3

#ENTER INTO BACK PROPAGATION PROCESS USING THE ERROR RECEIVED TO BACK INTO ALL UNITS WHILE UPDATING WEIGHTS.
# We are using deriaviative function of sigmoid for this case

#l3 delta value
l3_delta = self.sigmoid_derivative_ftn(l3) * target_pred_errors
#layer 2 error value
l2_error = l3_delta.dot(self.w2.T)
#get the delta fro layer 2
l2_delta  = self.sigmoid_derivative_ftn(l2) * l2_error

#using the delta values to update weights
self.w2 += self.learning_rate * l2.T.dot(l3_delta)
self.w1 += self.learning_rate * l1.T.dot(l2_delta)
```

The learning_rate variable is a hyperparameter that controls the magnitude of the weight updates. Larger values can lead to faster convergence, but may also cause the algorithm to overshoot the optimal weights. Smaller values can lead to slower convergence but more stable updates. After updating the weights, We repeat this process for the number of epochs defined by us (you can also set a threshold for where the loss need to reach). At each epoch, we print out the performance of the model using the score function, which calculates the accuracy of the model.

We can now run the simulation by creating a class Simulate_Backpropagation that will encompass everything and create an object with some sample data and running the fit method to train the model:

```python
# Define data to be used

# Input
X_data = np.array([[0, 0, 0], [1, 0, 0],  [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],  [0, 1, 1], [1, 1, 1]])

# Target output
Y_data = np.array([[0], [1], [1], [0], [0], [1], [1], [0]])


# Define Trainer Object
trainer = Simulate_Backpropagation(hidden_units =3, max_epochs=1000, learning_rate=0.9)
# Train the simulator
trainer.fit(X_data, Y_data)


# Get predictions
Ypreds = trainer.predict(X_data)
Ypreds_probs, weighted_sum = trainer.predict_proba(X_data)

print(f"Acctual Expected Output  {Y_data[:, 0]}")
print(f"Predicted  Output  {Ypreds[:, 0]}")
print(f"Predicted  Probabilities  {Ypreds_probs[:, 0]}")

```

Conclusion
Backpropagation is a key algorithm used in deep learning to train neural networks. It involves propagating errors backwards from the output layer to the input layer, and using these errors to update the weights of the network. The process is repeated for a fixed number of epochs or until the performance on the training data reaches a satisfactory level. This article has provided an introduction to backpropagation, including its mathematical formulas and a Python implementation using the NumPy library. We have also included sample code that can be used to simulate the backpropagation algorithm and make predictions on new data.
