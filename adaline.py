import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn import datasets


def ADALINE_Online(X, W, tol_error, max_epochs, bias):
    epochs = 0
    errors = []

    while epochs < max_epochs:
        error = 0
        indices = list(range(X.shape[0]))
        shuffle(indices)

        for i in indices:
            # Calcular o sinal do neurônio (spike)
            V = np.dot(W, X[i]) + bias
            # Ativação Linear
            Y = V
            error += (D[i] - Y) ** 2
            # Atualizar os pesos sinápticos da rede para todos os exemplos (SEMPRE)
            W = W + (D[i] - Y) * X[i]
        # Calcular o erro quadrático médio da época (MSE) usando todas as instâncias
        mse = (1 / X.shape[0]) * error
        errors.append(mse)
        epochs += 1
        if mse < tol_error:
            break

    return W, errors


iris = datasets.load_iris()
X = iris.data[:, :2]

D = iris.target

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / (X_std + 1e-8)
X = np.hstack((X, np.ones((X.shape[0], 1))))

W = np.random.rand(X.shape[1])
bias = 1.0
tol_error = 0.001
max_epochs = 1000

final_weights, errors = ADALINE_Online(X, W, tol_error, max_epochs, bias)

plt.plot(range(len(errors)), errors)
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.title('Curva de Aprendizado do ADALINE para o Conjunto de Dados Iris')
plt.show()

print("Pesos finais:", final_weights)
