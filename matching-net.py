import theano
import theano.tensor as T

import numpy as np

import pygame
from pygame.locals import *

pygame.init()

screen = pygame.display.set_mode((28 * 10, 28 * 4))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class LSTM:
    def __init__(self, inp, hidden, embedding):
        self.inp = inp
        self.hidden = hidden
        self.embedding = embedding

        self.weights = {
                            "f:x" : init_weights([self.inp, self.hidden]),
                            "f:h" : init_weights([self.hidden, self.hidden]),

                            "i:x" : init_weights([self.inp, self.hidden]),
                            "i:h" : init_weights([self.hidden, self.hidden]),

                            "o:x" : init_weights([self.inp, self.hidden]),
                            "o:h" : init_weights([self.hidden, self.hidden]),

                            "c:x" : init_weights([self.inp, self.hidden]),
                            "c:h" : init_weights([self.hidden, self.hidden]),
                       }

    def get_weights(self):
        return [self.weights[key] for key in self.weights.keys()]

    def recurrence(self, embedding, prev_hidden, prev_cell):
        forget = T.nnet.sigmoid(T.dot(embedding, self.weights["f:x"]) +
                                T.dot(prev_hidden, self.weights["f:h"]))

        input_ = T.nnet.sigmoid(T.dot(embedding, self.weights["i:x"]) +
                                T.dot(prev_hidden, self.weights["i:h"]))

        output = T.nnet.sigmoid(T.dot(embedding, self.weights["o:x"]) +
                                T.dot(prev_hidden, self.weights["o:h"]))

        cell = T.mul(forget, prev_cell) + T.mul(input_, T.tanh(T.dot(embedding, self.weights["c:x"]) +
        T.dot(prev_hidden, self.weights["c:h"])))

        hidden = T.mul(output, cell)

        return hidden, cell

class bidirectionalLSTM:
    def __init__(self, inp, hidden):
        self.inp = inp
        self.hidden = hidden

        self.forward = LSTM(inp, hidden, 0)
        self.backward = LSTM(inp, hidden, 0)

    def recurrence(self, embedding, prev_hidden_f, prev_cell_f, prev_hidden_b, prev_cell_b):
        hidden_f, cell_f = self.forward.recurrence(embedding, prev_hidden_f, prev_cell_f)

        hidden_b, cell_b = self.backward.recurrence(embedding, prev_hidden_b, prev_cell_b)

        embedding_ = embedding + hidden_f + hidden_b
        embedding_ = embedding_.dimshuffle([1])

        return hidden_f, cell_f, hidden_b, cell_b, embedding_

class embedder:
    def __init__(self):
        self.weights = [init_weights([16, 1, 3, 3]),
                        init_weights([16, 16, 3, 3]),
                        init_weights([16 * 49, 100])]

    def forward(self, inp):
        conv1 = T.nnet.conv2d(inp, self.weights[0], subsample = (2, 2), border_mode = 'half')
        conv1_relu = T.nnet.relu(conv1)

        conv2 = T.nnet.conv2d(conv1_relu, self.weights[1], subsample = (2, 2), border_mode = 'half')
        conv2_relu = T.nnet.relu(conv2)

        flatten = conv2_relu.flatten(ndim = 2)

        fc1 = T.dot(flatten, self.weights[2])

        return fc1

def init_weights(shape):
    return theano.shared(np.array(np.random.randn(*shape) * 0.01).astype(np.float64))

def cosine(batch_embeddings, support_set_embeddings):
    # Compares a batch image embedding to all support set embeddings using cosine similarity

    # embedded_batch, batch is a slice out of embedded_batch, and is of size 100
    # support_set_embeddings is of size support_set_size x 100

    # batch_magnitudes is of size i x 1
    batch_magnitude = T.sqrt(T.sum(batch_embeddings ** 2, axis = 1, keepdims=True))

    # support_set_magnitudes is of size support_set_size
    support_set_magnitudes = T.sqrt(T.sum(support_set_embeddings.T ** 2, axis = 0, keepdims=True))

    magnitudes = T.dot(batch_magnitude, support_set_magnitudes)

    dot = T.dot(batch_embeddings, support_set_embeddings.T)

    similarity = dot / magnitudes

    return [similarity]

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def main():
    # Define model
    # images x channels x width x height
    support_set = T.tensor4()

    # batch x channel x width x height
    batch = T.tensor4()

    # batch x support_set_size
    targets = T.matrix()

    embedder_g = embedder()
    embedder_f = embedder()

    support_set_lstm = bidirectionalLSTM(100, 100)

    # embedded_support_set is of shape images x 100
    embedded_support_set = embedder_g.forward(support_set)

    # get LSTM perturbed embeddings of support set
    ([hf, cf, hb, cb, embeddings], _) = theano.scan(fn =            support_set_lstm.recurrence,
                                                    sequences = [embedded_support_set],
                                                    outputs_info = [T.zeros([1, 100]), T.zeros([1, 100]), T.zeros([1, 100]), T.zeros([1, 100]), None])

    # embeddings is of shape images x 100

    # embedded_batch is of shape batchsize x 100
    embedded_batch = embedder_f.forward(batch)

    cosine_similarity = cosine(embedded_batch, embeddings)[0]

    # cosine_similarity is of shape batchsize x support_set_size
    # softmax along the last axis

    predictions = T.nnet.softmax(cosine_similarity)
    loss = -T.mean(T.sum(T.log(predictions) * targets + T.log(1 - predictions) * (1 - targets), axis = -1))

    updates = RMSprop(cost = loss, params = embedder_g.weights + embedder_f.weights + support_set_lstm.forward.get_weights() + support_set_lstm.backward.get_weights())

    train = theano.function(inputs = [support_set, batch, targets], outputs = [loss, predictions], updates = updates)

    #match = theano.function(inputs = [support_set, batch], outputs = cosine_similarity, updates = updates, on_unused_input='ignore')

    images = mnist.train.images
    labels = mnist.train.labels

    # what labels can we use?
    inclusion = [0, 1, 2, 3, 4]     # matching net will only ever get to see these labels

    bank = {}
    for label in inclusion:
        bank[label] = []

    for i in range (len(labels)):
        if np.argmax(labels[i]) in inclusion:
            # add the image
            bank[np.argmax(labels[i])].append([images[i], labels[i]])

    for episode in range (10000):
        # sample a set of labels
        num_labels = np.random.randint(2, 6)

        label_set = np.random.choice(inclusion, size = num_labels, replace = False)

        support_set = []
        label_order = []
        # for each label in the task, we're going to sample an example for the support set
        for label in label_set:
            idx = np.random.randint(0, len(bank[label]))
            sample = bank[label][idx][0]

            support_set.append(sample)
            label_order.append(np.argmax(bank[label][idx][1]))

        support_set = np.array(support_set).reshape([num_labels, 1, 28, 28])

        # batchsize is 10
        batch = []
        targets = []
        for b in range (10):
            label_idx = np.random.randint(0, len(label_order))
            label = label_order[label_idx]

            idx = np.random.randint(0, len(bank[label]))
            batch.append(bank[label][idx][0])

            onehot = np.zeros([len(label_order)])
            onehot[label_idx] = 1

            targets.append(onehot)

        batch = np.array(batch).reshape([10, 1, 28, 28])
        targets = np.array(targets)

        loss, pred = train(support_set, batch, targets)
        print(loss, pred)

        pygame.event.get()

        if (episode % 200 == 0):
            screen.fill((0, 0, 0))

            # Display a batch test case
            for x in range (28):
                for y in range (28):
                    screen.set_at((x, y), int(batch[0][0][y][x] * 255))

            # Display all elements in the support set
            for s in range (support_set.shape[0]):
                for x in range (28):
                    for y in range(28):
                        screen.set_at((s * 30 + x, y + 30), int(support_set[s][0][y][x] * 255))

            # get argmax
            prediction = np.argmax(pred[0])

            # draw box around the predicted class
            pygame.draw.rect(screen, (100, 255, 100), (prediction * 30, 30, 28, 28), 2)

            pygame.display.flip()

    """
    support_set = images[0:5].reshape(5, 1, 28, 28)
    batch = images[6:20].reshape(14, 1, 28, 28)

    labels = np.zeros([14, 5])
    for j in range (14):
        labels[j][j % 5] = 1

    for i in range (1000):

        print(train(support_set, batch, labels))
    """

main()
