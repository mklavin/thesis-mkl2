import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymcr.constraints import ConstraintNonneg
from pymcr.mcr import McrAR
from pymcr.metrics import mse
from pymcr.regressors import OLS, NNLS

def forward_propagation(W, H):
    """
    Calculate the predicted matrix A_pred using matrix W and H.
    """
    return np.dot(W, H)

def frobenius_norm(matrix):
    """
    Calculate the Frobenius norm of the matrix.
    """
    return np.linalg.norm(matrix)

def NMF(matrix, learning_rate, epochs, k):
    """
    Perform Non-negative Matrix Factorization (NMF) using Stochastic Gradient Descent (SGD).
    """
    m, n = matrix.shape
    W = np.random.uniform(0, 1, size=(m, k))  # Initialize W with random values
    H = np.random.uniform(0, 1, size=(k, n))  # Initialize H with random values

    return gradients(matrix, W, H, learning_rate, epochs)

def gradients(matrix, W, H, learning_rate, epochs):
    """
    Perform SGD to update W and H for minimizing the Frobenius norm loss
    while enforcing non-negativity constraints.
    """
    for epoch in range(epochs):
        # Calculate the predicted matrix
        A_pred = forward_propagation(W, H)

        # Calculate the error
        error = matrix - A_pred

        # Calculate the Frobenius norm loss
        loss = frobenius_norm(error)

        # Update W and H using gradients with non-negativity constraints
        grad_W = -2 * np.dot(error, H.T)
        grad_H = -2 * np.dot(W.T, error)

        # Update W and H with learning rate
        W -= learning_rate * grad_W
        H -= learning_rate * grad_H


        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss {loss}')

    return W, H

if __name__ == '__main__':
    df = pd.read_csv('data/150gg_data_prepro.csv')

    U, sig, V = np.linalg.svd(df)

    W, H = NMF(np.array(df.T), .001, 100000, 15)

    print(len(W[:, 0]))
    print(H.shape)

    plt.plot(W[:, 0])
    plt.plot(W[:, 1])
    plt.plot(W[:, 2])
    plt.plot(W[:, 3])
    plt.show()

