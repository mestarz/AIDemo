import os
from keras.models import Sequential
from keras.layers import Dense, Input
import tensorflow as tf
from keras import backend as K

import configure

HUBER_LOSS_DELTA = 2.0


@tf.function
def huber_loss(y_true, y_predict):
    err = y_true - y_predict

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)

    return K.mean(loss)


class Brain(object):

    def __init__(self, state_size, action_size, brain_name, test=False):
        self.state_size = state_size
        self.action_size = action_size
        self.weight_backup = brain_name
        self.test = test
        self.num_nodes = configure.NUMBER_NODES
        self.model = self._build_model()
        self.model_ = self._build_model()

    def _build_model(self):

        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(self.num_nodes, activation='relu'))
        model.add(Dense(self.num_nodes, activation='relu'))
        model.add(Dense(self.num_nodes, activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))

        # model = Sequential({
        #     Input(shape=(self.state_size,)),
        #     Dense(self.num_nodes, activation='relu'),
        #     Dense(self.num_nodes, activation='relu'),
        #     Dense(self.num_nodes, activation='relu'),
        #     Dense(self.action_size, activation='linear')
        # })

        model.compile(loss=huber_loss, optimizer='rmsprop')

        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error:no file')
            else:
                model.load_weights(self.weight_backup)

        return model

    def train(self, x, y, sample_weight=None, epochs=1, verbose=0):  # x is the input to the network and y is the output

        self.model.fit(x, y, batch_size=len(x), sample_weight=sample_weight, epochs=epochs, verbose=verbose)

    def predict(self, state, target=False):
        if target:  # get prediction from target network
            return self.model_.predict(state)
        else:  # get prediction from local network
            return self.model.predict(state)

    def predict_one_sample(self, state, target=False):
        return self.predict(state.reshape(1, self.state_size), target=target).flatten()

    def update_target_model(self):
        self.model_.set_weights(self.model.get_weights())

    def save_model(self, prefix):
        self.model.save(prefix + self.weight_backup)
