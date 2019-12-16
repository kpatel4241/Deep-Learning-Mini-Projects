import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/' , validation_size=0)

# size of the encodeing layer
encoding_dimensions = 32

# input and target placeholders
image_size = mnist.train.images.shape[1]

inputs = tf.placeholder(tf.float32, (None, image_size), name='inputs')
targets = tf.placeholder(tf.float32, (None, image_size), name='targets')

# output of hidden layer
encoded = tf.layers.dense(inputs , encoding_dimensions , activation=tf.nn.relu)

# Output layer logits
logits = tf.layers.dense(encoded, image_size, activation=None)

# sigmoid output from
decoded = tf.nn.sigmoid(logits,name='output')


loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets , logits=logits)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# create the tensorflow session
tf_session = tf.Session()

epochs = 20
batch_size = 200

tf_session.run(tf.global_variable_initlizers())
for e in range(epochs):
    for i in range(mnist.train.num//batch_size):
        batch = mnist.train.next_batch(batch_size)
        feed = {inputs : batch[0] , targets : batch[0]}
        batch_cost , _ = tf_session.run([cost,optimizer] , feed_dict=feed)
        print("Epoch: {}/{}...".format(e + 1, epochs),"Training loss: {:.4f}".format(batch_cost))


# checkout the results
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs = mnist.test.images[:10]
reconstructed, compressed = sess.run([decoded, encoded], feed_dict={inputs_: in_imgs})

for images, row in zip([in_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)


tf_session.close()
