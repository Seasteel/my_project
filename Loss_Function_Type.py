import numpy as np

predictions = np.array([[0.25, 0.25, 0.25, 0.25],
                       [0.01, 0.01, 0.01, 0.96]])
targets = np.array([[0, 0, 0, 1],
                    [0, 0, 0, 1]])
def crossEntropy(predictions, targets, epsilon=1e-10):
    predictions = np.clip(predictions, epsilon, 1. -epsilon)
    n = predictions.shape[0]
    ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5))) / n
    return ce_loss
cross_entropy_loss = crossEntropy(predictions, targets)
print('Cross entropy loss is: ' +str(cross_entropy_loss))

## RMSE loss function

y_hat = np.array([0.000, 0.166, 0.333])
y = np.array([0.000, 0.254, 0.998])

def mse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val
val = mse(y_hat, y)
print('mse error is: ' +str(val))

## MAE Loss function
y_hat = np.array([0.000, 0.166, 0.333])
y = np.array([0.000, 0.254, 0.998])

def mae(predictions, targets):
    differences = predictions - targets
    differences_absolute = np.absolute(differences)
    mean_differences_absolute = differences_absolute.mean()
    return mean_differences_absolute

mae_val = mae(y_hat, y)
print('mae error is: ' + str(mae_val))
