# -*- coding: utf-8 -*-

import timeit, cPickle

import numpy
import theano
import theano.tensor as T
import sys
import pickle
import nltk

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import random
random.seed(893)
# 67540
# 9876
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-lr','--learning_rate', help='learning rate, float', default=0.1, type=float, required=False)
parser.add_argument('-bs','--batch_size', help='batch size, integer', default=10, type=int, required=False)
parser.add_argument('-hl','--hidden_layer', help='hidden layer, integer', default=3000, type=int, required=False)
parser.add_argument('-tc','--target_class', help='target class', default='contingency', type=str, choices = ['temporal', 'contingency', 'comparison', 'extension'], required=False)
parser.add_argument('-pe','--polyglot_we', help='pickled file with word embeddings', type=str, required=True)
parser.add_argument('-df','--dataset', help='dataset file', type=str, required=True)

args = vars(parser.parse_args())

learning_rate = args['learning_rate']
batch_size = args['batch_size']
n_hidden = args['hidden_layer']
target = args['target_class']
polyglot = args['polyglot_we']
dataset = args['dataset']

def load_model():
    # From words to indices
    words, embeddings = pickle.load(open(polyglot, 'rb'))
    return embeddings, words

def load_dataset(mode="RUS", target=target):

    def knowledge_loader(knowledge):
        with open(knowledge, "r") as k:
            examples = []
            for i, l in enumerate(k):
                l = l.strip()
                args = l.split("\t")
                words1 = nltk.word_tokenize(args[0])
                words2 = nltk.word_tokenize(args[1])
                words1 = args[0].split(" ")
                words2 = args[1].split(" ")
                v1 = args[2]
                v2 = args[3]
                p1 = args[4].split(" ")
                p2 = args[5].split(" ")
                d1 = zip(*[d.split("|") for d in args[6].split(" ")])
                d2 = zip(*[d.split("|") for d in args[7].split(" ")])
                dv1 = list(d1[0] + d2[0])
                dv2 = list(d1[1] + d2[1])
                label = args[8]
                example = [["<S>"] + words1 + ["<S/>", "<S>"] + words2 + ["<S/>"], v1, v2, p1, p2, dv1, dv2, label]
                examples.append(example)
            return examples

    sets = knowledge_loader(dataset)

    train_length = int(len(sets) * 0.8)
    valid_length = int(len(sets) * 0.9)

    train_set = sets[:train_length]
    valid_set = sets[train_length : valid_length]
    test_set = sets[valid_length:]

    train_set_x = numpy.asarray(zip(*train_set)[0])
    train_set_y = numpy.asarray(zip(*train_set)[-1])
    valid_set_x = numpy.asarray(zip(*valid_set)[0])
    valid_set_y = numpy.asarray(zip(*valid_set)[-1])
    test_set_x = numpy.asarray(zip(*test_set)[0])
    test_set_y = numpy.asarray(zip(*test_set)[-1])

    train_set_z = zip(*train_set)[1:-1]
    valid_set_z = zip(*valid_set)[1:-1]
    test_set_z = zip(*test_set)[1:-1]

    words = load_model()[1]
    w2i = {words[i]:i for i in range(len(words))}

    # converting to indices
    def idx_mapper(x):
        idxs = [w2i.get(w.lower(), 3) for w in x]
        return idxs

    train_set_x = [idx_mapper(x) for x in train_set_x]
    valid_set_x = [idx_mapper(x) for x in valid_set_x]
    test_set_x = [idx_mapper(x) for x in test_set_x]

    # padding indices with -1
    longest = len(max(train_set_x + valid_set_x + test_set_x, key=len)) + 1
    def pad_element(e, length):
        if len(e) >= length:
            e = e[:length]
        elif len(e) < length:
            e += [-1] * (length - len(e))
        return e

    train_set_x = [pad_element(e, longest) for e in train_set_x]
    valid_set_x = [pad_element(e, longest) for e in valid_set_x]
    test_set_x = [pad_element(e, longest) for e in test_set_x]

    # verb indexing
    v1itr = [idx_mapper([x]) for x in train_set_z[0]]
    v2itr = [idx_mapper([x]) for x in train_set_z[1]]
    v1iv = [idx_mapper([x]) for x in valid_set_z[0]]
    v2iv = [idx_mapper([x]) for x in valid_set_z[1]]
    v1ite = [idx_mapper([x]) for x in test_set_z[0]]
    v2ite = [idx_mapper([x]) for x in test_set_z[1]]

    # participant indexing and padding
    p1itr = [idx_mapper(x) for x in train_set_z[2]]
    p2itr = [idx_mapper(x) for x in train_set_z[3]]
    p1ptr = [pad_element(e, 3) for e in p1itr]
    p2ptr = [pad_element(e, 3) for e in p2itr]

    p1iv = [idx_mapper(x) for x in valid_set_z[2]]
    p2iv = [idx_mapper(x) for x in valid_set_z[3]]
    p1pv = [pad_element(e, 3) for e in p1iv]
    p2pv = [pad_element(e, 3) for e in p2iv]

    p1ite = [idx_mapper(x) for x in test_set_z[2]]
    p2ite = [idx_mapper(x) for x in test_set_z[3]]
    p1pte = [pad_element(e, 3) for e in p1ite]
    p2pte = [pad_element(e, 3) for e in p2ite]

    # Unzipping distances and padding
    d1tr = [pad_element(x, longest) for x in train_set_z[4]]
    d2tr = [pad_element(x, longest) for x in train_set_z[5]]
    d1v = [pad_element(x, longest) for x in valid_set_z[4]]
    d2v = [pad_element(x, longest) for x in valid_set_z[5]]
    d1te = [pad_element(x, longest) for x in test_set_z[4]]
    d2te = [pad_element(x, longest) for x in test_set_z[5]]

    # Combining features
    train_set_z = zip(*[v1itr, v2itr, p1ptr, p2ptr, d1tr, d2tr])
    valid_set_z = zip(*[v1iv, v2iv, p1pv, p2pv, d1v, d2v])
    test_set_z = zip(*[v1ite, v2ite, p1pte, p2pte, d1te, d2te])

    train_set_z = [e[0] + e[1] + e[2] + e[3] + e[4] + e[5] for e in train_set_z]
    valid_set_z = [e[0] + e[1] + e[2] + e[3] + e[4] + e[5] for e in valid_set_z]
    test_set_z = [e[0] + e[1] + e[2] + e[3] + e[4] + e[5] for e in test_set_z]

    train_set_xz = numpy.concatenate([numpy.array(train_set_x, "int32"), numpy.array(train_set_z, "int32")], axis=1)

    train_set_xzy = zip(train_set_xz, train_set_y)
    posex = [e for e in train_set_xzy if int(e[1]) == 1]
    negex = [e for e in train_set_xzy if int(e[1]) == 0]
    negex = random.sample(negex, len(posex))
    train_set_xzy = posex + negex
    random.shuffle(train_set_xzy)
    train_set_xz, train_set_y = zip(*train_set_xzy)
    train_set_x = numpy.asarray(train_set_xz)[:, :longest]
    train_set_z = numpy.asarray(train_set_xz)[:, longest:]

    # make data and labels shared
    def shared_dataset(x, z, y,  borrow=True):
        shared_x = theano.shared(numpy.asarray(x, dtype='int32'), borrow=borrow)
        shared_z = theano.shared(numpy.asarray(z, dtype='int32'), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_z, T.cast(shared_y, 'int32')

    train_set_x, train_set_z, train_set_y = shared_dataset(train_set_x, train_set_z, train_set_y)
    valid_set_x, valid_set_z, valid_set_y = shared_dataset(valid_set_x, valid_set_z, valid_set_y)
    test_set_x, test_set_z, test_set_y = shared_dataset(test_set_x, test_set_z, test_set_y)

    return train_set_x, train_set_z, train_set_y, valid_set_x, valid_set_z, valid_set_y, test_set_x, test_set_z, test_set_y, longest

class WordEmbedding(object):

    def __init__(self, input, We, features, longest):

        """
        Input = a list (minibatch) of lists of indexes (pre-processed sentences)
        We = a word embedding matrix (vocabulary * dimensions)
        """

        # initialise the word embeddings
        self.We = theano.shared(numpy.asarray(We, dtype=theano.config.floatX), name='We', borrow=True)
        # Mapping to vector:
        # from: input (batch_size * indices)
        # to: vectors (batch_size * indices * dimensions)
        lookup = self.We[input]
        # This step concatenates along the vector-dimension axis three versions of the lookup 3D tensor:
        # 1) lookup 'rolled' forwards by 1 along the indices axis, 2) original tesor 3) tensor shifted backwards by 1
        # Note that in 1) and 3) a 0-valued vector represent sentence boundaries.
        forwards = T.set_subtensor(T.roll(lookup, 1, axis=1)[:,0], 0.)
        backwards = T.set_subtensor(T.roll(lookup, -1, axis=1)[:,-1], 0.)
        window_processing = T.concatenate([forwards, lookup, backwards], axis=2)

        event1 = self.We[features[:, 0]]
        event2 = self.We[features[:, 1]]
        participants1 = T.max(self.We[features[:, 2:5]], axis=1)
        participants2 = T.max(self.We[features[:, 5:8]], axis=1)
        # Lexical features n_examples * dimensions
        lex = T.concatenate([event1, event2, participants1, participants2], axis=1)

        positions1 = features[:, 8:8+longest, numpy.newaxis] / 100.
        positions2 = features[:, 8+longest:, numpy.newaxis] / 100.
        senpos = T.concatenate([window_processing, positions1, positions2], axis=2)

        # I/O
        self.input = input
        self.output = [senpos, lex]
        # parameters of the model
        self.params = [self.We]

class HiddenLayer(object):
    def __init__(self, input, rng, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Units are fully-connected but have no non-linear activation function.
        Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        senpos = input[0]
        lex = input[1]

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)


        if b is None:
            b = theano.shared(
            value=(numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX)
            ),
            name='b',
            borrow=True
        )

        self.W = W
        self.b = b

        # Max operation over indices

        lin_output = T.dot(senpos, self.W) + self.b
        # Non-linear activation function is tanh, max operation over indices
        nonlin_output = T.max(activation(lin_output), axis=1)
        # Concatenation of sentence+positional with lexical features
        shalconv= T.concatenate([nonlin_output, lex], axis=1)
        self.output = shalconv / shalconv.norm(L=2)
	# parameters of the model
        self.params = [self.W, self.b]
        self.input = input

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for softmax
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of max probability
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        # Parameter dimensionality for normalization
        self.L2 = (self.W ** 2).sum()

    def xent(self, y):

        return T.nnet.categorical_crossentropy(self.p_y_given_x, y).mean()

    def errors(self, y):
        """Zero-one loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return round((f1_score(y, self.y_pred)), 2)

class NN(object):
    """Neural Network Class stacking a Word Embedding, two Hidden Layers and Linear Regression"""

    def __init__(self, rng, input, We, n_in, n_hidden, n_out, longest, features):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # WordEmbeddings
        self.wordEmbedding = WordEmbedding(
            input=input,
            We=We,
            longest=longest,
            features=features

        )

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=self.wordEmbedding.output,
            n_in=n_in,
            n_out=n_hidden
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden + 256,
            n_out=n_out
        )

        self.L1 = abs(self.wordEmbedding.We).sum() + abs(self.hiddenLayer.W).sum() \
            + abs(self.logRegressionLayer.W).sum()

        self.L2_sqr = (self.wordEmbedding.We).sum() + (self.hiddenLayer.W ** 2).sum() \
            + (self.logRegressionLayer.W ** 2).sum()

        # cross-entropy of the NN is given by the cross-entropy of the output of the model
        self.xent = (self.logRegressionLayer.xent)
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layers
        self.params = self.wordEmbedding.params + self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = input

        # keep track of predictions
        self.y_pred = self.logRegressionLayer.y_pred

def train_class(learning_rate=learning_rate, L1_reg=0.00, L2_reg=0.0001, n_epochs=200, batch_size=batch_size, n_hidden=n_hidden):
    """
    Demonstrate stochastic gradient descent optimization for a NN

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

   """


    print("Feature Type:\tenriched\nBatch Size:\t{}\nHidden Layer:\t{}\nLearning Rate:\t{}\nTarget Class:\t{}\n".format(batch_size, n_hidden, learning_rate, target))

    We = load_model()[0]

    ######################
    # LOAD DATASET #
    ######################
    print('... loading the dataset\n')

    train_set_x, train_set_z, train_set_y, valid_set_x, valid_set_z, valid_set_y, test_set_x, test_set_z, test_set_y, longest = dataset_base

    # compute number of minibatches for training and validation
    n_train_batches = train_set_x.eval().shape[0] // batch_size
    n_valid_batches = valid_set_x.eval().shape[0] // batch_size
    n_test_batches = test_set_x.eval().shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model\n')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.imatrix('x')  # the data
    z = T.imatrix('z')  # the features
    y = T.ivector('y')  # the labels

    rng = numpy.random.RandomState(25158)

    # construct the NN class

    classifier = NN(
        rng=rng,
        input=x,
        We=We,
        n_in=194,
        n_hidden=n_hidden,
        n_out=2,
        longest=longest,
        features=z
    )

    # the cost we minimize during training is the cross-entropy of
    # the model plus the regularization terms (L1 and L2)
    cost = (
        classifier.xent(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=[y, classifier.y_pred],
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            z: valid_set_z[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    test_model = theano.function(
        inputs=[index],
        outputs=[y, classifier.y_pred],
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            z: test_set_z[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function that returns the cost and updates the parameter of the model
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            z: train_set_z[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        })

    ################
    # TRAIN MODEL #
    ###############
    print('... training the model\n')
    best_validation_loss = 0.
    start_time = timeit.default_timer()
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    epoch = 0
    bestest_y = None
    while (epoch < n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            # minibatch cost
            train_loss = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % n_train_batches == 0:
                # compute zero-one loss on validation set
                validation_losses = [f1_score(*validate_model(i))
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation f1-score %f %%\n' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                gold_y, best_y = zip(*[t for t in test_losses])
                gold_y = [item for sublist in gold_y for item in sublist]
                best_y = [item for sublist in best_y for item in sublist]
                test_f1 = f1_score(gold_y, best_y)
                test_p = precision_score(gold_y, best_y)
                test_r = recall_score(gold_y, best_y)
                test_a = accuracy_score(gold_y, best_y)
                print(('     epoch %i, minibatch %i/%i, test f1-score %f %%, precision %f %%, recall %f %%, accuracy %f %%.\n'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_f1 * 100.,
                            test_p * 100.,
                            test_r * 100.,
                            test_a * 100.
                        )
                    )

                sys.stdout.flush()
                # if we got the best validation score until now
                if this_validation_loss > best_validation_loss * improvement_threshold:
                    best_validation_loss = this_validation_loss
                    bestest_y = best_y
                    # save the best model
                    with open('feature_bs-{}_hl-{}_lr{}_t-{}.pkl'.format(batch_size, n_hidden, learning_rate, target), 'wb') as f:
                        cPickle.dump((classifier.params[0].get_value(), classifier.params[1].get_value(), classifier.params[2].get_value(),
                                      classifier.params[3].get_value(), classifier.params[4].get_value()), f, protocol=cPickle.HIGHEST_PROTOCOL)


    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% ') %
          (best_validation_loss * 100.))
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    print(bestest_y)

dataset_base = load_dataset()
train_class()
