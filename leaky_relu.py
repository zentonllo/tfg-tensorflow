import tensorflow as tf
# Obtenido de Tensorlayer
def leaky_relu(x=None, alpha=0.1, name="LeakyReLU"):
    """The LeakyReLU, Shortcut is ``lrelu``.

    Modified version of ReLU, introducing a nonzero gradient for negative
    input.

    Parameters
    ----------
    x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
        `int16`, or `int8`.
    alpha : `float`. slope.
    name : a string or None
        An optional name to attach to this activation function.

    Examples
    ---------
    >>> network = tl.layers.DenseLayer(network, n_units=100, name = 'dense_lrelu',
    ...                 act= lambda x : tl.act.lrelu(x, 0.2))

    References
    ------------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models, Maas et al. (2013) <http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf>`_
    """
    with tf.name_scope(name):
        x = tf.maximum(x, alpha * x)
    return x
