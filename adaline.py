import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

def ADALINE_Online(X, W, tol_error, max_epochs, bias, learning_rate):
    epochs = 0
    errors = []

    while epochs < max_epochs:
        error = 0
        indices = list(range(X.shape[0]))
        shuffle(indices)

        for i in indices:
            V = np.dot(W, X[i]) + (bias * W[-1])
            Y = V
            error += (D[i] - Y) ** 2
            W = W + learning_rate * (D[i] - Y) * X[i]

        mse = (1 / X.shape[0]) * error
        errors.append(mse)
        epochs += 1
        if mse < tol_error:
            break

    return W, errors

iris = datasets.load_iris()
X = iris.data[:, :2]
D = iris.target

# Normalização das características usando StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.hstack((X, np.ones((X.shape[0], 1))))

# Inicialização dos pesos mais próximos de zero
W = np.array([0.01, 0.02, 0.03])

bias = 1.0
tol_error = 0.0001
max_epochs = 100  # Alterando max_epochs para 100
learning_rate = 0.0001  # Alterando learning_rate para 0.0001

final_weights, errors = ADALINE_Online(X, W, tol_error, max_epochs, bias, learning_rate)

plt.plot(range(len(errors)), errors)
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.title('Curva de Aprendizado do ADALINE para o Conjunto de Dados Iris')
plt.show()

print("Pesos finais:", final_weights)
