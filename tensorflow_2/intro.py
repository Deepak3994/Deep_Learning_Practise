import tensorflow as tf

# define the inuts
x = tf.constant(2.0)
y = tf.constant(8.0)

# define the graph
@tf.function
def my_function(x, y):
    g_mean = tf.sqrt(x*y)
    return g_mean


res = my_function(x, y)
print(res)

w = tf.Variable(tf.ones(shape=(2,2)), name="w")
b = tf.Variable(tf.ones(shape=(2)), name="b")

@tf.function
def forward(x):
    return w * x + b


out_a = forward([1,0])
print(out_a)
