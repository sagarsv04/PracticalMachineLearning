import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
# working with the MNIST dataset, which is a dataset that
# contains 60,000 training samples and 10,000 testing samples
# of hand-written and labeled digits, 0 through 9, so ten total "classes."
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

'''
# one_hot means
# 0 = [1,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0]
# 2 = [0,0,1,0,0,0,0,0,0]
# 3 = [0,0,0,1,0,0,0,0,0]
# ...
'''


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# input data
x = tf.placeholder('float', [None, 784])
# label of data
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    # first layer taking data as input
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    # passing it to relu activation function
    l1 = tf.nn.relu(l1)

    # second layer taking first layer's output as input
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    # passing it to relu activation function
    l2 = tf.nn.relu(l2)

    # third layer taking second layer's output as input
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    # passing it to relu activation function
    l3 = tf.nn.relu(l3)

    # output layer taking third layer's output as input
    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    # returns list consisting value ranging from 0-1 for each member in list [1,2,3,4,5,6,7,8,9]
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # old version:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # new version:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # adaptive moment estimation.
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # number of cycle
    hm_epochs = 10

    with tf.Session() as sess:
        # old:
        # sess.run(tf.initialize_all_variables())
        # new:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # _ is a variable we don't care about
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        return 0


def main():
    train_neural_network(x)
    return 0


if __name__ == '__main__':
    main()
