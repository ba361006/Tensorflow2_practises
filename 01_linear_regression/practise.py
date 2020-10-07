from tensorflow import keras
import tensorflow as tf 
import numpy as np 
import os 

# to ignore the annoying messages from running with tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Network(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.add_weight("weight",shape=[1], dtype=tf.float32)
        self.bias = self.add_weight("bias",shape=[1], dtype=tf.float32)
    
    def call(self, input_metrices):
        output = input_metrices * self.weight + self.bias
        return output

# create 100 data randomly
x_data = tf.random.uniform([100], dtype=tf.float32)
x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

# this equation is our target equation
y_data = x_data * 2.7 + 3.5

# covert data into tf.data.Data
dataset = tf.data.Dataset.from_tensor_slices((x_data,y_data))

# build the network, here we don't need to model.build because we don't keras.Sequential
model = Network()
optimizer = keras.optimizers.Adam(learning_rate=1e-2)

# start training
for epoch in range(201):
    for step, (xs,answer) in enumerate(dataset):
        with tf.GradientTape() as tape:
            ys = model(xs)

            """ don't know why, but this doesn't work
            # MSE = keras.losses.MeanSquaredError()
            loss = MSE(answer, ys)
            """
            loss = keras.losses.MSE(answer, ys)
            
        gardient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gardient, model.trainable_variables))

    if epoch % 20 == 0:
        print("------------------------------------------------")
        print("Epoch: {}, loss: {}".format(epoch, loss))
        print("predicted weight: {}, predicted bias: {}".format(model.weight.numpy(), model.bias.numpy()))
        



