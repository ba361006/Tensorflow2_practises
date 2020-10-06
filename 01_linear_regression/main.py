import tensorflow as tf
import numpy as np
from tensorflow import keras
import  os


class Regressor(keras.layers.Layer):

    def __init__(self):
        super(Regressor, self).__init__()

        # here must specify shape instead of tensor !
        # name here is meanless !
        # [dim_in, dim_out]
        self.w = self.add_variable('meanless_name', [13, 1])

        # [dim_out]
        self.b = self.add_variable('meanless_name', [1])

        print(self.w.shape, self.b.shape)
        print("type(self.w): {}, tf.is_tensor: {}, self.w.name: {}".format(type(self.w), tf.is_tensor(self.w), self.w.name))
        print("type(self.b): {}, tf.is_tensor: {}, self.b.name: {}".format(type(self.b), tf.is_tensor(self.b), self.b.name))

    # the function call is from the parent class -> keras.layers.Layer
    # for further information, see line 152~154 in the defination of keras.layers.Layer
    def call(self, x):
        x = tf.matmul(x, self.w) + self.b
        return x

def main():
    # set the global seed, we get different results for every call to the random op
    # but the same sequence for every re-run of the program
    tf.random.set_seed(22)

    # same situation as here
    np.random.seed(22)

    # ignore 0: INFO, 1: WARNING, 2:ERROR, 3:FETAL messages, it needs to be single quote
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # raise AssertionError if tf version isn't 2.
    assert tf.__version__.startswith('2.')


    # reading data
    # x_train.shape, y_train.shape, x_val.shape, y_val.shape 
    # (404, 13),     (404,)       , (102, 13)  , (102,)
    (x_train, y_train), (x_val, y_val) = keras.datasets.boston_housing.load_data()
    x_train, x_val = x_train.astype(np.float32), x_val.astype(np.float32)
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(102)

    model = Regressor()
    loss_function = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)

    for epoch in range(200):

        for step, (x, y) in enumerate(db_train):
            # x.shape: (64,13); y.shape: (,64)
            with tf.GradientTape() as tape:
                # [b, 1]
                output = model(x)

                # [b]
                # flatten output
                flatten = tf.squeeze(output, axis=1)

                # [b] vs [b]
                loss = loss_function(y, flatten)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # print(epoch, 'loss:', loss.numpy())


        if epoch % 10 == 0:

            for x, y in db_val:
                # [b, 1]
                output = model(x)
                # [b]
                output = tf.squeeze(output, axis=1)
                # [b] vs [b]
                loss = loss_function(y, output)

                # print(epoch, 'val loss:', loss.numpy())





if __name__ == '__main__':
    main()