import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def mnist_dataset():
    (x,y), (x_val, y_val) = datasets.mnist.load_data()
    
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)
    y = tf.one_hot(y, depth=10)
    y_val = tf.one_hot(y_val, depth=10)
    
    dataset_train = tf.data.Dataset.from_tensor_slices((x,y))
    dataset_train = dataset_train.shuffle(60000).batch(100)

    dataset_test = tf.data.Dataset.from_tensor_slices((x_val,y_val))
    dataset_test = dataset_test.shuffle(60000).batch(100)

    return dataset_train, dataset_test


def main():
    ds_train, ds_test = mnist_dataset()

    model = keras.Sequential([layers.Reshape(target_shape=(28*28,), input_shape=(28,28)),
                              layers.Dense(200, activation='relu'),
                              layers.Dense(200, activation='relu'),
                              layers.Dense(200, activation='relu'),
                              layers.Dense(10)])
                              
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])                          

    model.fit(ds_train.repeat(), epochs=30, steps_per_epoch=500,
              validation_data=ds_test.repeat(),
              validation_steps=2)

if __name__ == "__main__":
    main()

