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

# Keras
> For further information, [check here](htt:ps//towardsdatascience.com/how-to-calculate-the-number-of-parameters-in-keras-models-710683dae0ca).

<br>

- Number of parameter for Convolutional Neural Network:<br>

    <center>

    ```
    num_param = output_channel_number * (input_channel_number * kernel_height * kernel_width + 1)
    ```
    </center>

    Note: The number 1 denotes the bias that is associated with each filter that we’re learning.

    <br>

    - Eample :<br>

        There is a Conv2D layer, <br>
        <center>
        
        | Input shape | Kernel size | output shape |
        | :---: | :---: | :---: |
        | (3,3) | (28,28,1) | (26,26,32) |
        </center>

        and its number of trainable parameters would be <br>
        <center> 320 = 32 * (1 * 3 * 3 + 1) </center>
    
- Number of parameter for fully connected layer:
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
