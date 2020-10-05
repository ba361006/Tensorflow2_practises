import  tensorflow as tf
import numpy as np 
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


xs = np.linspace(1,54,54).reshape(6,3,3)
print("\nxs.shape: ", xs.shape)
print("xs: ", xs)

xs = tf.convert_to_tensor(xs)
print("\ntf.convert_to_tensor: ", xs)

xs = tf.data.Dataset.from_tensor_slices(xs)
print("\ntf.data.Dataset.from_tensor_slices: ", xs)

# number of batch would be "channel of xs / xs.batch" -> 6 / 3 = 2
xs = xs.batch(3)
print("\nxs.batch(3): ", list(xs.as_numpy_iterator()))

xs = xs.repeat(3)
print("\nxs.repeat(3): ", list(xs.as_numpy_iterator()))

# In this example: 
# epoch: 6
# batch size: 3
# iteration / number of batch: 2

