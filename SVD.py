import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg
from scipy.linalg import svd
from numpy.linalg import matrix_power
from pymcr.constraints import ConstraintNonneg
from pymcr.mcr import McrAR
from pymcr.metrics import mse
from pymcr.regressors import OLS, NNLS

def create_diagonal_matrix(values, m):
    # Create a diagonal matrix with the given values
    diagonal_matrix = np.diag(values)

    # Check if m is greater than the number of values
    if m > len(values):
        # Compute the number of additional rows
        num_zero_rows = m - len(values)
        # Add additional rows with zeros
        zero_rows = np.zeros((num_zero_rows, len(values)))
        # Concatenate zero rows to the diagonal matrix
        diagonal_matrix = np.vstack((diagonal_matrix, zero_rows))

    return diagonal_matrix

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

def eq2(U, W, learning_rate, epochs, k):
    """
    Perform Non-negative Matrix Factorization (NMF) using Stochastic Gradient Descent (SGD).
    """
    m, n = U.shape

    B = np.random.uniform(0, 1, size=(k, n))  # Initialize H with random values

    return calc_B(U, W, B, learning_rate, epochs)

def calc_B(U, W, B, learning_rate, epochs):

    """
        Perform SGD to update W and H for minimizing the Frobenius norm loss
        while enforcing non-negativity constraints.
        """
    for epoch in range(epochs):

        # Calculate the predicted matrix
        A_pred = forward_propagation(W, B)

        # Calculate the error
        error = U - A_pred

        # Calculate the Frobenius norm loss
        loss = frobenius_norm(error)

        # Update B using gradients with non-negativity constraints
        grad_B = -2 * np.dot(W.T, error)

        # Update B with learning rate
        B -= learning_rate * grad_B

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss {loss}')

    return B

def eq_3(B, E, H):

    ans = np.dot(B, E)

    ans = np.linalg.inv(ans)

    return np.dot(ans, H)

if __name__ == '__main__':
    df = pd.read_csv('data/150gg_data_prepro_GSH.csv')

    U, E, Vt = svd(df.T)
    E = create_diagonal_matrix(E, U.shape[1])

    W, H = NMF(np.array(df.T), .0002, 10000, 77)

    B = eq2(U, W, .0001, 10000, 77)

    ans = eq_3(B, E, H)

    plt.plot(ans[0])
    plt.show()







    # plt.plot(W[:, 0])
    # plt.plot(W[:, 1]+5)
    # plt.plot(W[:, 2]+10)
    # plt.plot(W[:, 3]+15)
    # plt.plot(W[:, 4]+20)
    # plt.plot(W[:, 5]+25)
    # plt.plot(W[:, 6]+30)
    # plt.plot(W[:, 7]+35)
    # plt.plot(W[:, 8] +40)
    # plt.plot(W[:, 9] + 45)
    # plt.plot(W[:, 10] + 50)
    # plt.plot(W[:, 11] + 55)
    # plt.plot(W[:, 12] + 60)
    # plt.plot(W[:, 13] + 65)
    # plt.plot(W[:, 14] + 70)
    # plt.show()

