import tensorflow as tf




class RegressionNN(tf.keras.Model):
    """
    Class which implements a neural network for solving a regression
    task.
    """
    def __init__(self, n_units) -> None:
        super(RegressionNN, self).__init__()
        if isinstance(n_units, int):
            n_units = [n_units]
        
        self._layers = []
        
        for n in n_units:
            self._layers.append(tf.keras.layers.Dense(n, activation='relu'))
        self._layers.append(tf.keras.layers.Dense(1))

    def call(self, input):
        x = input
        for layer in self._layers:
            x = layer(x)

        return x



def get_model(n_units):
    """
    """
    if isinstance(n_units, int):
        n_units = [n_units]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((11,)))

    for n in n_units:
        model.add(tf.keras.layers.Dense(n, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model