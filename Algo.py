import numpy as np


class Simulate_Backpropagation:
    """
    THis class Object simulate the working of a Model Unit with Backpropagation to update weigghts.
    It starts by randomly generating weights and then using these to optise them for the model using backpropation method
    """
    def __init__(self, hidden_units=3, learning_rate=0.01, max_epochs=10):
        self.hidden_units = hidden_units
        #set random seed
        np.random.seed(0)
        #initialize weights based on hidden units
        self.w1 =  np.random.rand(3, self.hidden_units)
        self.w2 =  np.random.rand(self.hidden_units, 1)
        #define learning rate
        self.learning_rate = learning_rate
        # number of epochs
        self.max_epochs = max_epochs




    # sigmoid function
    def sigmoid_ftn(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative for sigmoid ftn
    def sigmoid_derivative_ftn(self,x):
        return x * (1 - x)

    #function to train the model
    def fit(self, X, Y):
        print(f"Started training........")
        print(f"Initial Hidden Layyer Weights:\n{self.w1}")
        print(f"Inital Output Layer Weights  {list(trainer.w2[:,0])}\n")
        #we go it iteration from 0 to max_epoch times while trainingthe network
        for i in range(self.max_epochs):
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

            if i%100 == 0:
                #print(f"Hidden Layyer Weights:\n   {[list(x) for x in self.w1]}")
                #print(f"Output Layer Weights  {list(self.w2[:,0])}")
                print(f"Epoch {i}  Perfomance is  {self.score(X,Y)}")

        print("\nFinal Weights")
        print(f"Final Hidden Layyer Weights:\n{self.w1}")
        print(f"Final Output Layer Weights  {list(self.w2[:,0])}")
        print()

    #preds probs or sum of weights
    def predict_proba(self, X):
        weighted_sum = np.dot(self.sigmoid_ftn(np.dot(X, self.w1)), self.w2)
        return self.sigmoid_ftn(weighted_sum), weighted_sum
    #function to get new preditions
    def predict(self, X):
        #It is 1 if prediction Value is over 0.9 else 0
        preds = self.sigmoid_ftn(np.dot(self.sigmoid_ftn(np.dot(X, self.w1)), self.w2))
        return np.array([[1] if x >=0.9 else [0] for x in preds])

    #functiion to the trainer perfomance
    def score(self, X, Y):
        return f"Accuracy:  {(self.predict(X) == Y).sum()/len(Y) * 100}%"



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
print(f"Predicted  Probs Through Sigmoind  {Ypreds_probs[:, 0]}")
print(f"Predicted  weighted sum Output  {weighted_sum[:, 0]}")



# get the score
print(trainer.score(X_data, Y_data))


# test with some inputs
print(f" Prediction for  [1, 1, 0] is   {trainer.predict([1, 1, 0])[:,0]}")



print(f"Final Hidden Layyer Weights:\n{trainer.w1}")
