# Note

- main.py
    > revised from [04-Linear-Regression](https://github.com/dragen1860/TensorFlow-2.x-Tutorials/tree/master/04-Linear-Regression)

    This programme is to build a linear regression network for Boston_housing price prediction

    - Boston Housing Dataset

        The medv variable is the target variable.

        ![image of Boston_housing](http://wordpress.wbur.org/wp-content/uploads/2013/09/housing-photo1-1000x664.jpg)
        Credits: http://www.wbur.org/radioboston/2013/09/18/bostons-housing-challenge

        <div align="center">

        | Feature | Description |
        | ------- | ----------- |
        | crim    | per capita crime rate by town.  |
        | zn      | proportion of residential land zoned for lots over 25,000 sq.ft. |
        | indus   | proportion of non-retail business acres per town. |
        | chas    | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise). |
        | nox     | nitrogen oxides concentration (parts per 10 million). |
        | rm      | average number of rooms per dwelling. |
        | age     | proportion of owner-occupied units built prior to 1940. |
        | dis     | weighted mean of distances to five Boston employment centres. |
        | rad     | index of accessibility to radial highways. |
        | tax     | full-value property-tax rate per \$10,000. |
        | ptratio | pupil-teacher ratio by town. |
        | black   | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town. |
        | lstat   | lower status of the population (percent). |
        | medv    | median value of owner-occupied homes in \$1000s. |
        
        </div>
    

- practise.py
    
    This programme builds a linear regression model for equation prediction

    Equation: 

    <div align="center">y_data = x_data * 2.7 + 3.5</div>

    <br>
    You can also change the batch size, learning_rate and optimizer to see how them affacts the system by the following lines

    
    
    ```python
    # change the batch size
    dataset = tf.data.Dataset.from_tensor_slices((x_data,y_data)).batch(10)
    ```

    ```python
    # For changing the optimizer, you might need to adjest the number of epoch
    optimizer = keras.optimizers.SGD(learning_rate=1e-2)
    ```
    

        