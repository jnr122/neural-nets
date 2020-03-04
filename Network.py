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

        # store sizes and instances of weights and biases
        self.w_sizes = [(x.shape[1],ls), (self.ls, self.ls), (self.ls, y.shape[1])]
        self.b_sizes = [(1,self.ls), (1, self.ls), (1, y.shape[1])]
        self.ws = []
        self.bs = []

        # weights and biases for layers
        for w, b in zip(self.w_sizes, self.b_sizes):
            self.ws.append(np.random.randn(w[0], w[1]))
            self.bs.append(np.zeros((b[0], b[1])))


    # training loop
    def train(self):
        for x in range(self.epochs):
            self.feed()
            self.back()
            if x % 100 == 0:
                print('Error for epoch ' + str(x) + ': ' + str(self.loss))

    def feed(self):
        z1 = np.dot(self.x, self.ws[0]) + self.bs[0]   # feed layer 1
        self.a1 = sig(z1)
        z2 = np.dot(self.a1, self.ws[1]) + self.bs[1]  # feed layer 2
        self.a2 = sig(z2)
        z3 = np.dot(self.a2, self.ws[2]) + self.bs[2]  # feed output

        # get normalized max of output layer
        self.a3 = softmax(z3)

    def back(self):
        self.loss = error(self.a3, self.y)

        a3_delta = cross(self.a3, self.y)        # adjustments for layer 3->2
        z2_delta = np.dot(a3_delta, self.ws[2].T)
        a2_delta = z2_delta * dsig(self.a2)      # adjustments for layer 2->1
        z1_delta = np.dot(a2_delta, self.ws[1].T)
        a1_delta = z1_delta * dsig(self.a1)      # adjustments for layer 1

        # tuning for layers
        self.ws[2] -= self.lr * np.dot(self.a2.T, a3_delta)
        self.bs[2] -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.ws[1] -= self.lr * np.dot(self.a1.T, a2_delta)
        self.bs[1] -= self.lr * np.sum(a2_delta, axis=0)
        self.ws[0] -= self.lr * np.dot(self.x.T, a1_delta)
        self.bs[0] -= self.lr * np.sum(a1_delta, axis=0)

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
def sig(outs): return 1 / (1 + np.exp(-outs))          # sigmoid
def dsig(outs): return outs * (1 - outs)               # sigmoid deriv
def cross(pre, act): return (pre - act) / act.shape[0] # compute error by cost over num elements


# normalize the outputs of the output layer so the outputs
# represent confidence by probability
def softmax(outs):
    exps = np.exp(outs - np.max(outs, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# for the sake of printing progress
def error(pre, act):
    sample_size = act.shape[0]
    logp = - np.log(pre[np.arange(sample_size), act.argmax(axis=1)])
    loss = np.sum(logp) / sample_size
    return loss
#
#    END AUXILIARY FUNCS
#############################
