# Glossary

- Epoch :<br>
The process of a dataset is passed throught the neural network once and returned once

- Batch :<br>
When the data cannot be passed through the neural network at one time, the dataset needs to be divided into several batches

- Batch Size :<br>
Number of samples passed through the neural network at once

- Iteration :<br>
How many batches that is needed to complete an epoch

- Example :<br> 
There is a dataset of 2000 training samples. Divide 2000 samples into batches of 500, so 4 samples are needed to complete an epoch.

<center>

Epoch | Batch | Batch Size | Iteration
:---: | :---: | :---: | :---:
1 | 4 | 500 | 4

</center>


<br>

# Notes
> For further information, [check here](htt:ps//towardsdatascience.com/how-to-calculate-the-number-of-parameters-in-keras-models-710683dae0ca).

<br>

### Demonstarion
- tf.data.Dataset: 
    ```
    from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
    import tensorflow as tf
    import numpy as np 
    import os


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    xs = np.linspace(1,12,12).reshape(3,2,2)
    print("\nxs.shape: ", xs.shape)
    print("xs: \n", xs)

    xs = tf.convert_to_tensor(xs)
    print("------------------------------------------------------------")
    print("tf.convert_to_tensor: \n", xs)

    xs = tf.data.Dataset.from_tensor_slices(xs)
    print("------------------------------------------------------------")
    print("Note:")
    print("If we print xs in this stage, it will return the first batch, pay attention to the shape")
    print("\ntf.data.Dataset.from_tensor_slices: ", xs)

    # number of batch would be "channel of xs / xs.batch" -> 6 / 3 = 2
    xs = xs.batch(3)
    print("------------------------------------------------------------")
    print("Note:")
    print("Dataset has been divided into 2 group, and has 3 data in each group, ")
    print("which means that batch size would be 3, and the iteration / number of batch would be 2")
    print("\nxs.batch(3): ", list(xs.as_numpy_iterator()))

    xs = xs.repeat(3)
    print("------------------------------------------------------------")
    print("Note:")
    print("Copy dataset three times")
    print("\nxs.repeat(3): ", list(xs.as_numpy_iterator()))
    ```

- tf.GradientTape():

    Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. Tensors can be manually watched by invoking the watch method on this context manager.

    > for further information, check the official website [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape)

    ```
    import tensorflow as tf 
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    x = tf.constant(3, dtype=tf.float32)

    # with GradientTape()
    with tf.GradientTape() as tape:
        tape.watch(x)
        y1 = x*x

    # Differentiate with respect to x
    # let y = x^2 -> f'(x) = dy/dx = 2x -> f'(3) = 6
    dy_dx = tape.gradient(y1, x)
    print("dy_dx: ", dy_dx)
    ```

<br>

### Counting the number of parameter
- Convolutional Neural Network:<br>

    <center>

    ```
    num_param = output_channel_number * (input_channel_number * kernel_height * kernel_width + 1)
    ```
    </center>

    Note: The number 1 denotes the bias that is associated with each filter that we’re learning.

    <br>

    - Eample :<br>

        There is a Conv2D layer <br>
        <center>
        
        | Input shape | Kernel size | output shape |
        | :---: | :---: | :---: |
        | (3,3) | (28,28,1) | (26,26,32) |
        </center>

        its number of trainable parameters would be <br>
        <center> 320 = 32 * (1 * 3 * 3 + 1) </center>
    
- fully connected layer:
    <center>

    ```
    num_param = neurons_in_flatten_layer * neurons_in_fully_connected + num_of_bias
                                    or
    num_param = (neurons_in_flatten_layer + 1) * neurons_in_fully_connected
    ```
    </center>
    
    Note: Number of bias would equal to number of neurons_in_fully_connected.

    - Example :
        <center>
        
        | neurons_in_flatten_layer | neurons_in_fully_connected | num_of_bias |
        | :---: | :---: | :---: |
        | 576 | 64 | 64 |
        </center>

        and its number of trainable parameters would be <br>
        <center> 36928 = 576 * 64 + 64 </center>

