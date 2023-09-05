import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


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


# Exemplo com uma porta lógica AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
W = np.random.rand(X.shape[1])
D = np.array([0, 0, 0, 1])
bias = 1.0
tol_error = 0.001
max_epochs = 1000

final_weights, errors = ADALINE_Online(X, W, tol_error, max_epochs, bias)

plt.scatter(range(len(D)), D, label='Esperado', marker='o', color='blue')
plt.scatter(range(len(D)), np.dot(X, final_weights) +
            bias, label='Obtido', marker='x', color='red')
plt.xlabel('Amostras')
plt.ylabel('Saída')
plt.title('ADALINE - Porta lógica AND')
plt.legend()
plt.show()

print("Pesos finais:", final_weights)
