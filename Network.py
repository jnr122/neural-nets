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
        self.loss = 0
        self.connections = 3

        # store sizes and instances of weights and biases
        self.w_sizes = [(self.x.shape[1],ls), (self.ls, self.ls), (self.ls, self.y.shape[1])]
        self.b_sizes = [(1, self.ls), (1, self.ls), (1, self.y.shape[1])]
        self.ws = []
        self.bs = []

        # weights and biases for layers
        for w, b in zip(self.w_sizes, self.b_sizes):
            self.ws.append(np.random.randn(w[0], w[1]))
            self.bs.append(np.zeros((b[0], b[1])))

        # unactivated/ activated/ deltas shapes
        self.zouts = self.aouts = ([np.zeros((self.x.shape[0], self.bs[0].shape[1]))] * self.connections)
        self.dzs = self.das = ([np.zeros(self.x.shape)] * self.connections)

    # training loop
    def train(self):
        for x in range(self.epochs):
            self.feed()
            self.back()
            if x % 100 == 0:
                print('Error for epoch ' + str(x) + ': ' + str(self.loss))

    # feed forward
    def feed(self):
        for i in range(self.connections):
            self.zouts[i] = np.dot(self.x, self.ws[i]) + self.bs[i] if (i == 0) else np.dot(self.aouts[i-1], self.ws[i]) + self.bs[i]
            self.aouts[i] = sig(self.zouts[i]) if (i != self.connections-1) else sig(self.zouts[i])

    # backprop
    def back(self):
        self.loss = error(self.aouts[self.connections-1], self.y)

        adjustments = [[]] * 3

        # calculate gradients
        # self.das[self.connections-1] = cross(self.aouts[self.connections-1], self.y)
        for i in range(self.connections-1, -1, -1):
            if i == self.connections-1:
                self.das[i] = cross(self.aouts[i], self.y)
            else:
                self.dzs[i] = np.dot(self.das[i+1], self.ws[i+1].T)
                self.das[i] = self.dzs[i] * dsig(self.aouts[i])
            if i != 0:
                adjustments[i] = self.lr * np.dot(self.aouts[i-1].T, self.das[i]), self.lr * np.sum(self.das[i], axis=0)
            else:
                adjustments[i] = self.lr * np.dot(self.x.T, self.das[0]),  self.lr * np.sum(self.das[0], axis=0)

        # adjust
        for w,b,a in zip(self.ws, self.bs, adjustments):
            w -= a[0]
            b -= a[1]

    def predict(self, data):
        self.x = data
        self.feed()
        return self.aouts[self.connections-1].argmax()
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
