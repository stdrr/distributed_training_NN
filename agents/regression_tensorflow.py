from agents.agent import Agent
import tensorflow as tf



class RidgeRegressionTF(Agent):
    """
    Class which solves a distributed regression task with squared error
    and L2 regularization (Ridge Regression).
    """
    def __init__(self, model:tf.keras.Model, data, batch_size, lambda_, alpha, alpha_decay, name):
        data = data.map(self._preprocess_data).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        super(RidgeRegressionTF, self).__init__(model, data)
        self._batch_size = batch_size
        self._lambda_ = lambda_
        self._alpha = alpha
        self._alpha_decay = alpha_decay
        self.test_history = []
        self.name = name

    
    def _preprocess_data(self, x, y):
        """
        """
        x = [tf.cast(t, tf.float32) for t in x.values()]
        return tf.stack(x), tf.cast(y, dtype=tf.float32)


    def _compute_gradients(self, w):
        """
        """
        n_batches = tf.cast(self._data.cardinality(), dtype=tf.float32)
        mse = tf.keras.losses.MeanSquaredError()
        for x_batch, y_batch, in self._data:
            with tf.GradientTape() as tape:
                y_pred = self._model(x_batch)
                loss_g = mse(y_batch, y_pred) 
                accum_vars = [tf.Variable(tf.zeros_like(tv.value()), trainable=False) for tv in w]
                zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
            grads_batch = tape.gradient(loss_g, w)
            grads_g = [accum_vars[i].assign_add(grad / n_batches) for i, grad in enumerate(grads_batch)]

        return grads_g


    def _compute_w_tilde(self, grads_g, pi_tilde):
        """
        """
        w_tilde = []

        for grad_j, pi_j in zip(grads_g, pi_tilde):
            w_tilde_j = (-1 / self._lambda_) * (grad_j + pi_j)
            w_tilde.append(w_tilde_j)
        
        return w_tilde

    
    def _compute_z(self, w, w_tilde):
        """
        """
        z = []

        for w_j, w_tilde_j in zip(w, w_tilde):
            z_j = w_j + self._alpha * (w_tilde_j - w_j)
            z.append(z_j)
        
        return z


    def _compute_w(self, c, z_neighbors):
        """
        :param c: weight coefficients for the links of the agent
        :param z_neighbors: parameters of the neighbors; [(param1_n1, ..., param1_nm), (param2_n1, ..., param2_nm), ...]
        """
        w = []

        for z_neighbors_j in z_neighbors:

            w_j = tf.zeros_like(z_neighbors_j[0])
            for z_j, c_j in zip(z_neighbors_j, [self.name] + self._endpoints):
                w_j += c[c_j] * z_j

            w.append(w_j)
        
        return w

    
    def _compute_y(self, c, y_neighbors, grads_g, grads_g_prev):
        """
        """
        y = []

        for y_neighbors_j, grad_g, grad_g_prev in zip(y_neighbors, grads_g, grads_g_prev):

            y_j = tf.zeros_like(y_neighbors_j[0])
            for y_neighbor_j, c_j in zip(y_neighbors_j, [self.name] + self._endpoints):
                y_j += c[c_j] * y_neighbor_j
            
            y_j += grad_g - grad_g_prev
            y.append(y_j)

        return y


    def _compute_pi_tilde(self, y, grads_g, I):
        """
        """
        pi_tilde = []

        for y_j, grad_g in zip(y, grads_g):
            pi_tilde_j = I * y_j - grad_g
            pi_tilde.append(pi_tilde_j)

        return pi_tilde

    
    def _get_neighbors_params(self, y, z):
        """
        """
        params_y, params_z = [y], [z]

        # print(f'Process {self.name}: Getting parameters')
        
        for endpoint_name in self._endpoints:
            p = self._monitor.read_shared_resource(endpoint_name, self.name)
            params_y.append(p[0])
            params_z.append(p[1])
        
        params_y = zip(*params_y)
        params_z = zip(*params_z)

        # print(f'Process {self.name}: Got parameters ({len(y)}, {len(z)})')

        return params_y, params_z

    
    def _send_params_to_neighbors(self, y, z):
        """
        """
        y_numpy = [y_j.numpy() for y_j in y]
        z_numpy = [z_j.numpy() for z_j in z]
        # print(f'Process {self.name}: Sending parameters')
        self._monitor.write_shared_resource(self.name, (y_numpy,z_numpy))
        # print(f'Process {self.name}: Sent parameters ({len(y)}, {len(z)})')

        
    def train(self, epochs, c, I, test_data):
        """
        :param epochs: maximum number of epochs
        :param c: weight coefficients for the links of the agent
        """
        w = self._model.trainable_weights
        grads_g = self._compute_gradients(w)
        y = grads_g
        pi_tilde = self._compute_pi_tilde(y, grads_g, I)

        # Step 1: iterate until stop criterion is met
        for i in range(1, epochs+1):

            # Step 2.
            w_tilde = self._compute_w_tilde(grads_g, pi_tilde)
            z = self._compute_z(w, w_tilde)

            # Step 3.
            self._send_params_to_neighbors(y, z)
            y_neighbors, z_neighbors = self._get_neighbors_params(y, z)
            self._model.set_weights(self._compute_w(c, z_neighbors))
            w = self._model.trainable_weights
            grads_g_prev = grads_g
            grads_g = self._compute_gradients(w)
            y = self._compute_y(c, y_neighbors, grads_g, grads_g_prev)
            pi_tilde = self._compute_pi_tilde(y, grads_g, I)
            self._alpha = self._alpha_decay(self._alpha)

            if test_data is not None:
                self.test_history.append(self.evaluate(test_data, self._batch_size))

        print(f'Node {self.name}: End of training after {i} epochs')
        self._monitor.put_return_value(self.name, self.test_history)
        return self.test_history
        

    def evaluate(self, test_data, batch_size=32):
        """
        """
        test_data = test_data.map(self._preprocess_data).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        mse = tf.keras.metrics.MeanSquaredError()
        for x_batch, y_batch in test_data:
            y_pred = self._model(x_batch, training=False)
            mse.update_state(y_batch, y_pred)
        result = mse.result().numpy()
        self.test_history.append(result)
        return result
