import  tensorflow as tf
import os
import numpy as np 
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

# load data
(xs, ys),_ = datasets.mnist.load_data()
print("\nxs.shape: ", xs.shape)
print("\nys.shape: ", ys.shape)

# convert data as tensor
xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.

# input tensor into tf.data.Dataset
db = tf.data.Dataset.from_tensor_slices((xs,ys))

# for further information in this step, check remark.py 
db = db.batch(32).repeat(10)


# build a network
network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(256, activation='relu'),
                      layers.Dense(256, activation='relu'),
                      layers.Dense(10)])

network.build(input_shape=(None, 28*28))

# show the information about params
network.summary()

# select Sochastic Gradient Descent optimizer from keras.optimizer
optimizer = optimizers.SGD(lr=0.01)

# compute accuracy
acc_meter = metrics.Accuracy()


for step, (x,y) in enumerate(db):
    # shape of x = (32,28,28), due to db.batch(32)
    # shape of y = (32,)
    with tf.GradientTape() as tape:
        # [b, 28, 28] -> [b, 784]
        x = tf.reshape(x, (-1, 28*28))

        # [b, 784] -> [b, 10]
        out = network(x)

        # [b] -> [b, 10]
        # tf.one_hot(3, depth=10) -> [0,0,1,0,0,0,0,0,0,0]
        y_onehot = tf.one_hot(y, depth=10)

        # [b, 10]
        loss = tf.square(out-y_onehot)

        # [b]
        # divide 32 here to get the mean of loss
        loss = tf.reduce_sum(loss) / 32

    # update acc_meter by inputing result and label
    acc_meter.update_state(tf.argmax(out, axis=1), y)

    grads = tape.gradient(loss, network.trainable_variables)

    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    if step % 200==0:
        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())

        # initalize acc_meter, otherwise it will keep acccumulating the result from beginning
        acc_meter.reset_states()