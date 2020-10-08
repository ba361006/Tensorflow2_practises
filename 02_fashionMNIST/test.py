import tensorflow as tf 

a = tf.data.Dataset.range(10).repeat()
print(list(a.as_numpy_iterator()))