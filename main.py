import os
import numpy as np
import pandas as pd
from threading import Thread

# Loading training data.
data = pd.read_csv("mnist_train.csv")
data = np.array(data)
x, y = [], []
on = False
correct, total = 0, 0
for arr in data:
    x.append(arr[1:])
    y.append(arr[0])

# Loading testing data.
data = pd.read_csv("mnist_test.csv")
data = np.array(data)
x_test, y_test = [], []
for arr in data:
    x_test.append(arr[1:])
    y_test.append(arr[0])


# Initializing weights and biases.
def Initialize(a1, a2):
    w = np.random.randn(a2, a1)
    b = np.random.rand(a2)
    return w, b


# Applying weights and biases to the neurons, then applying an activation function.
def layer(a, w, b, activation):
    result = 'error'
    if activation == "ReLU":
        result = ReLU(np.dot(w, a) + b)
    if activation == "softmax":
        result = softmax(np.dot(w, a) + b)
    return result


# Normalizing data to get values between 0 and 1.
def normalize(inp):
    return inp / max(inp)


def ReLU(a):
    return np.maximum(0, a)


def deriv_ReLU(a):
    return a > 0


def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))


# Manually one hot encoding the expected output (don't ask why).
def expected_array(Y):
    array = np.array([])
    for n in range(10):
        if n != Y:
            array = np.append(array, 0)
        else:
            array = np.append(array, 1)
    return array


# I forgot the purpose of this.
def multiply(a, a2):
    result = []
    for val in a:
        result.append(val * a2)
    return np.array(result)


# Updating the weights and biases using back-propagation.
def backprop(W1, B1, W2, B2, A2, A1, X, Y, r):
    dZ2 = A2 - Y
    dW2 = multiply(dZ2, A1)
    dB2 = np.sum(dZ2)
    dZ = np.dot(w2.T, dZ2) * deriv_ReLU(A1)
    dW = multiply(dZ, X)
    dB = np.sum(dZ)
    return W1 - r * dW, B1 - r * dB, W2 - r * dW2, B2 - r * dB2


# The model training function.
def run():
    global w1, w2, b1, b2, correct, total
    correct, total = 0, 0
    while True:
        if on:
            n = np.random.randint(0, 10000)
            x_work, y_work = normalize(x[n]), y[n]
            layer1 = layer(x_work, w1, b1, "ReLU")
            output_layer = layer(layer1, w2, b2, "softmax")

            expected = expected_array(y_work)
            w1, b1, w2, b2 = backprop(w1, b1, w2, b2, output_layer, layer1, x_work, expected, 0.05)
            if list(output_layer).index(np.max(output_layer)) == y_work:
                correct += 1
            total += 1


# The model testing function.
def modelTest(n):
    global w1, w2, b1, b2, correct, total
    correct, total = 0, 0
    x_work, y_work = normalize(x_test[n]), y_test[n]
    layer1 = layer(x_work, w1, b1, "ReLU")
    output_layer = layer(layer1, w2, b2, "softmax")
    print("The number:", y_work, "|| The model predicted:", list(output_layer).index(np.max(output_layer)))


# Commands for friendly usage.
def commands():
    global w1, b1, w2, b2, on
    while True:
        try:
            command = input("Console> ")
            if command == "/accuracy":
                print("Accuracy: ", correct / total)
            elif command == "/start testing":
                while True:
                    command = input("Chose an index (/stop for stop)> ")
                    if command == "/stop":
                        print("Stopped testing.")
                        break
                    else:
                        try:
                            modelTest(int(command))
                        except:
                            print('is not included')
            elif command == "/start training":
                if on:
                    print("The model is already getting trained.")
                else:
                    print("The model started!")
                    on = True
            elif command == "/stop training":
                if on:
                    print("The model stopped!")
                    on = False
                else:
                    print("The model isn't running.")
            elif "/save" in command:
                path = command.replace("/save ", "")
                os.mkdir(path)
                np.savetxt(path + "/w1.w", w1)
                np.savetxt(path + "/w2.w", w2)
                np.savetxt(path + "/b1.b", b1)
                np.savetxt(path + "/b2.b", b2)
                print("Model saved!")
            elif "/load" in command:
                path = command.replace("/load ", "")
                w1, w2, b1, b2 = np.loadtxt(path + "/w1.w"), np.loadtxt(path + "/w2.w"), np.loadtxt(path + "/b1.b"), np.loadtxt(path + "/b2.b")
                print("Weights and Biases imported successfully.")
            else:
                print("Unknown command...")
        except:
            print("error")


com = Thread(target=commands)
if __name__ == '__main__':
    w1, b1 = Initialize(784, 16)
    w2, b2 = Initialize(16, 10)
    com.start()
    run()
