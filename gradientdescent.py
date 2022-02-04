import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import noisy

dim_theta = 10
data_num = 1000
scale = .001

theta_true = np.ones((dim_theta,1))
print('True theta:', theta_true.reshape(-1))

A = np.random.uniform(low=-1.0, high=1.0, size=(data_num,dim_theta))
y_data = A @ theta_true + np.random.normal(loc=0.0, scale=scale, size=(data_num, 1))

A_test = np.random.uniform(low=-1.0, high=1.0, size=(50, dim_theta))
y_test = A_test @ theta_true + np.random.normal(loc=0.0, scale=scale, size=(50, 1))
theta_pred = la.inv((A.transpose() @ A)) @ (A.transpose() @  y_data)
print('Empirical theta', theta_pred.reshape(-1))

batch_size = 1
max_iter = 1000
lr = 0.001
theta_init = np.random.random((10,1)) * 0.1

# Parameteres
deg_ =  2
num_rep =  10
max_iter =  1000
fig, ax = plt.subplots(figsize=(10,10))
best_vals = {}
test_exp_interval =  20
grad_artificial_normal_noise_scale = 0. 

for method_idx, method in enumerate(['adam', 'sgd', 'adagrad']):
    test_loss_mat = []
    train_loss_mat = []

    for replicate in range(num_rep):
        if replicate % 20 == 0:
            print(method, replicate)

        if method == 'adam':
            beta_1 = 0.9
            beta_2 = 0.999
            # 1st moment vector
            m = np.zeros_like(theta_init)
            # second moment vector
            v = np.zeros_like(theta_init)
            epsilon = 10 ** -8

        if method == 'adagrad':
            epsilon = 10 ** -8
            squared_sum = 0

        theta_hat = theta_init.copy()
        test_loss_list = []
        train_loss_list = []

        for t in range(max_iter):
            idx = np.random.choice(data_num, batch_size)  # Split data
            train_loss, gradient = noisy.noisy_val_grad(theta_hat, A[idx, :], y_data[idx, :], deg_=deg_)
            artificial_grad_noise = np.random.randn(10, 1) * grad_artificial_normal_noise_scale + np.sign(
                np.random.random((10, 1)) - 0.5) * 0.
            gradient = gradient + artificial_grad_noise
            train_loss_list.append(train_loss)

            if t % test_exp_interval == 0:
                test_loss, _ = noisy.noisy_val_grad(theta_hat, A_test[:, :], y_test[:, :], deg_=deg_)
                test_loss_list.append(test_loss)

            if method == 'adam':
                m = beta_1 * m + (1 - beta_1) * gradient
                v = beta_2 * v + (1 - beta_2) * (gradient ** 2)
                m_hat = m / (1 - (beta_1 ** (t + 1)))
                v_hat = v / (1 - (beta_2 ** (t + 1)))
                theta_hat = theta_hat - lr * m_hat / (np.sqrt(v_hat) + epsilon)

            elif method == 'adagrad':
                squared_sum = squared_sum + gradient ** 2
                theta_hat = theta_hat - lr * gradient / (np.sqrt(squared_sum + epsilon))

            elif method == 'sgd':
                theta_hat = theta_hat - lr * gradient

        test_loss_mat.append(test_loss_list)
        train_loss_mat.append(train_loss_list)
        print(method, replicate)
        print(theta_hat.reshape(-1))

    print(method, 'done')
    x_axis = np.arange(max_iter)[::test_exp_interval]

    test_loss_np = np.array(test_loss_mat)

    test_loss_mean = np.mean(test_loss_np, axis=0) 

    test_loss_se = np.std(test_loss_np, axis=0) / np.sqrt(num_rep) 

    plt.errorbar(x_axis, test_loss_mean, yerr=2.5 * test_loss_se, label=method)
    plt.title(f'Test Loss \n(objective degree: {deg_})')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.xlabel('Updates')
    best_vals[method] = min(test_loss_mean)
plt.show()