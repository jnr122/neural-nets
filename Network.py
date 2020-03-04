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

        # unactivated/ activated outs
        outs_shape = np.zeros((self.x.shape[0], self.bs[0].shape[1]))
        self.zouts = [outs_shape] * self.connections
        self.aouts = [outs_shape] * self.connections

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
        self.loss = error(self.aouts[2], self.y)

        a3_delta = cross(self.aouts[2], self.y)         # gradient for layer 3->2
        z2_delta = np.dot(a3_delta, self.ws[2].T)
        a2_delta = z2_delta * dsig(self.aouts[1])       # gradient for layer 2->1
        z1_delta = np.dot(a2_delta, self.ws[1].T)
        a1_delta = z1_delta * dsig(self.aouts[0])       # gradient for layer 1->in

        # calculate weight and bias adjustments
        adjustments = [(self.lr * np.dot(self.x.T, a1_delta),  self.lr * np.sum(a1_delta, axis=0)),
                       (self.lr * np.dot(self.aouts[0].T, a2_delta), self.lr * np.sum(a2_delta, axis=0)),
                       (self.lr * np.dot(self.aouts[1].T, a3_delta), self.lr * np.sum(a3_delta, axis=0))]

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
