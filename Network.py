import numpy as np


#############################
#    START NETWORK IMPLEMENTATION
#
class Net:
    def __init__(self, x, y, epochs, ls, lr):
        # data and hyperparams
        self.x = x
        self.y = y
        self.ls = ls
        self.lr = lr
        self.epochs = epochs

        # weights and biases for layers
        self.w1 = np.random.randn(x.shape[1], self.ls)
        self.b1 = np.zeros((1, ls))
        self.w2 = np.random.randn(self.ls, self.ls)
        self.b2 = np.zeros((1, self.ls))
        self.w3 = np.random.randn(self.ls,  y.shape[1])
        self.b3 = np.zeros((1,  y.shape[1]))

    # training loop
    def train(self):
        for x in range(self.epochs):
            self.feed()
            self.back()
            if x % 100 == 0:
                print('Error for epoch ' + str(x) + ': ' + str(self.loss))

    def feed(self):
        z1 = np.dot(self.x, self.w1) + self.b1   # feed layer 1
        self.a1 = sig(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2  # feed layer 2
        self.a2 = sig(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3  # feed output

        # get normalized max of output layer
        self.a3 = softmax(z3)

    def back(self):
        self.loss = error(self.a3, self.y)

        a3_delta = cross(self.a3, self.y)        # adjustments for layer 3->2
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * dsig(self.a2)      # adjustments for layer 2->1
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * dsig(self.a1)      # adjustments for layer 1

        # tuning for layers
        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feed()
        return self.a3.argmax()
#
#    END NETWORK IMPLEMENTATION
#############################


#############################
#    START AUXILIARY FUNCS
#
def sig(s): return 1 / (1 + np.exp(-s)) # sigmoid
def dsig(s): return s * (1 - s)          # sigmoid deriv
def cross(pre, act): return (pre - act) / act.shape[0] # compute error by cost over num elements


# normalize the outputs of the output layer so the outputs
# represent confidence by probability
def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# for the sake of printing progress
def error(pre, act):
    n_samples = act.shape[0]
    logp = - np.log(pre[np.arange(n_samples), act.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss
#
#    END AUXILIARY FUNCS
#############################