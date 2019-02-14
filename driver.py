from sklearn.datasets import load_boston
import numpy as np

from learner import Learner
from utils.stats import normalize, partition_data
from models.utils.loss import mse

boston = load_boston()

regularization = 0.0
lr = 1e-4
dr = 1e-4
me = 1e4
k = 30

learner = Learner()
sgd=dict(learning_rate=lr, decay_rate=dr, max_epoches=me)

# learner.gradient_checker(regularization, 'linear', perturbations=100, step_size=1e-4)

X = normalize(np.matrix(boston.data))
Y = np.matrix(boston.target).T
train_X, train_Y, test_X, test_Y = partition_data(X, Y, 0.8)

predict, training_err = learner.train(train_X, train_Y, 'linear', exact=True, k=k,
                                     plot=False, regularization=regularization)
Y_hat = predict(test_X)
print "Training error: %f, testing error: %f" % (training_err, mse(Y_hat, test_Y))
learner.plot("Testing results", observations=test_Y, predictions=Y_hat)

predict, training_err = learner.train(train_X, train_Y, 'linear', exact=False, k=k,
                                      plot=False, regularization=regularization, **sgd)
Y_hat = predict(test_X)
print "Training error: %f, testing error: %f" % (training_err, mse(Y_hat, test_Y))
learner.plot("Testing results", observations=test_Y, predictions=Y_hat)
