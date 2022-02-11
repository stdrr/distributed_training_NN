from agent import Agent
import tensorflow as tf



class RidgeRegressionTF(Agent):
    """
    Class which solves a distributed regression task with squared error
    and L2 regularization (Ridge Regression).
    """
    def __init__(self, model:tf.kears.Model, data, batch_size, lambda_, alpha):
        super(RidgeRegressionTF, self).__init__(model, data)
        self._batch_size = batch_size
        self._lambda = lambda_
        self._alpha = alpha


    def _compute_gradients(self, w):
        """
        """
        grads_g = [tf.zeros_like(param.value()) for param in self._model.trainable_weights]
        n_batches = tf.cast(self._data.cardinality(), dtype=tf.float32)
        for x_batch, y_batch in self._data:
            with tf.GradientTape() as tape:
                y_pred = self._model(x_batch)
                loss_g = tf.math.squared_difference(y_batch, y_pred) 
            grads_batch = tape.gradients(loss_g, w)
            for i in range(len(grads_g)):
                grads_g += grads_batch[i] / n_batches
        return grads_g


    def _compute_w_tilde(self, grads_g, pi_tilde):
        """
        """
        w_tilde = []

        for grad_j, pi_j in zip(grads_g, pi_tilde):
            w_tilde_j = (-1 / self._lambda) * (grad_j + pi_j)
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
            for z_j, c_j in zip(z_neighbors_j, c):
                w_j += c_j * z_j

            w.append(w_j)
        
        return w

    
    def _compute_y(self, c, y_neighbors, grads_g, grads_g_prev):
        """
        """
        y = []

        for y_neighbors_j, grad_g, grad_g_prev in zip(y_neighbors, grads_g, grads_g_prev):

            y_j = tf.zeros_like(y_neighbors_j)
            for y_neighbor_j, c_j in zip(y_neighbors_j, c):
                y_j += c_j * y_neighbor_j
            
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
        
        for endpoint_name in self._endpoints.keys():
            p = self._endpoints[endpoint_name].recv()
            params_y.append(p[0])
            params_z.append(p[1])
        
        params_y = zip(*params_y)
        params_z = zip(*params_z)

        return params_y, params_z

    
    def _send_params_to_neighbors(self, y, z):
        """
        """
        for endpoint_name in self._endpoints.keys():
            self._endpoints[endpoint_name].send((y,z))

        
    def train(self, epochs, c, **kwargs):
        """
        :param epochs: maximum number of epochs
        :param c: weight coefficients for the links of the agent
        """
        I = kwargs['n_agents']
        w = self._model.trainable_weights
        grads_g = self._compute_gradients(w)
        y = grads_g
        pi_tilde = self._compute_pi_tilde(y, grads_g, I)

        # Step 1: iterate until stop criterion is met
        for _ in range(epochs):

            # Step 2.
            w_tilde = self._compute_w_tilde(grads_g, pi_tilde)
            z = self._compute_z(w, w_tilde)

            # Step 3.
            self._send_params_to_neighbors(y, z)
            y_neighbors, z_neighbors = self._get_neighbors_params(y, z)
            w = self._compute_w(c, z_neighbors)
            self._model.set_weights(w)
            grads_g_prev = grads_g
            grads_g = self._compute_gradients(w)
            y = self._compute_y(c, y_neighbors, grads_g, grads_g_prev)
            pi_tilde = self._compute_pi_tilde(y, grads_g, I)

        
