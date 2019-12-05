import numpy as np
from helper import load_data, get_test_data, get_train_data

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """X is N x D where each row is an example. Y is 1-dimensional of size N"""
        #the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            print(f"predicting..{i}")
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example

        return Ypred

# fetch the training data from CIFAR-10
Xtr, Ytr = get_train_data()
# fetch the test data from CIFAR-10
Xte, Yte = get_test_data()

print(f'Xtr:  {Xtr.shape}')
print(f'Ytr:  {Ytr.shape}')
print(f'Xte:  {Xte.shape}')
print(f'Yte:  {Yte.shape}')

#flatten out all images to be one_dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

print(f'Xtr_rows:  {Xtr_rows.shape}')
print(f'Xte_rows:  {Xte_rows.shape}')

nn = NearestNeighbor()
nn.train(Xtr_rows, Ytr)
Yte_predict = nn.predict(Xte_rows)

accuracy = np.mean(Yte_predict == Yte)
print(f'accuracy: {accuracy}')