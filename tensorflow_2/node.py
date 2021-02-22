import tensorflow as tf


class Two_Feature_Node:
    def __init__(self):
        self.w = tf.Variable([[0.1],[0.2]])
        self.b = tf.Variable([[0.5]])

    def __call__(self, x):
        return self.get_output(x)

    def get_output(self, x):
        out = tf.matmul(x, self.w)
        out = tf.add(out, self.b)
        out = tf.math.sigmoid(out)
        return out


# two feature input
x = tf.constant([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]])
two_features_node = Two_Feature_Node()
print(two_features_node(x).numpy())