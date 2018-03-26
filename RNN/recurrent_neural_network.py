import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

'''
This a single layer recurrent neural net
With MNIST dataset the accuracy is above 97 %

'''


hm_epochs = 3
#  number of iteration
n_classes = 10
#  number of output classes
batch_size = 128
chunk_size = 28
n_chunks = 28

rnn_size = 128


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network(x):
    '''
    This is a single layer recurrent neural network

    '''
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    # LSTM cell that will recur for tthe rnn_size
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Every cell has output and states

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))
        #  one image at a time


def main():
    train_neural_network(x)
    return 0



if __name__ == '__main__':
    main()
