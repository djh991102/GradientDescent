import numpy as np
def noisy_val_grad(theta_hat, data_, label_, deg_=2.):
    gradient = np.zeros_like(theta_hat)
    loss = 0

    for i in range(data_.shape[0]):
        x_ = data_[i, :].reshape(-1, 1)
        y_ = label_[i, 0]
        err = np.sum(x_ * theta_hat) - y_

        grad = np.sign(err) * deg_ * x_ * np.abs(err) ** (deg_ - 1)
        l = np.abs(err) ** deg_

        loss += l / data_.shape[0]
        gradient += grad / data_.shape[0]

    return loss, gradient