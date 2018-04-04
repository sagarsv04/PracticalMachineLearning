import pydicom as dicom # for reading dicom files
import os # for doing directory operations
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import tensorflow as tf
import numpy as np

# Change this to wherever you are storing your data:
# IF YOU ARE FOLLOWING ON KAGGLE, YOU CAN ONLY PLAY WITH THE SAMPLE DATA, WHICH IS MUCH SMALLER

# os.getcwd()
# os.chdir(r'D:\CodeRepo\PracticalMachineLearning\Kaggle')

img_size_px = 128
slice_count = 20

data_dir = './sample_images/'
lable_csv = './stage1_labels.csv'

n_classes = 2
batch_size = 10
keep_rate = 0.8

x = tf.placeholder('float')
y = tf.placeholder('float')


def chunks(l, n):
    '''
    Yield successive n-sized chunks from l.
    Get n chuncks from list l.
    '''
    # l, n = slices, chunk_sizes
    for i in range(0, len(l), n):
        yield l[i:i + n]


def mean(l):
    '''
    Return mean of a list
    '''
    return sum(l) / len(l)


def data_handling(num_of_patient=1):

    patients = os.listdir(data_dir)
    labels_df = pd.read_csv(lable_csv, index_col=0)
    # labels_df.head()
    print("No. Of Patients::", len(patients))
    # for one patient get all the slices of scans
    for patient in patients[:num_of_patient]:
        # patient = patients[:1][0]
        label = labels_df.get_value(patient, 'cancer') # deprecated replace it with the new method to return label
        # label = labels_df.at(patient, 'cancer')
        path = data_dir + patient
        # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
        # to get all the slices from directory of patients
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        # sorting the slices in the list with respect to image position
        # slices[0].ImagePositionPatient[2]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        # print(slices[0])
        print("Image Size::{0}, No. Of Slices::{1}, Label::{2}".format(slices[0].pixel_array.shape,len(slices),label))


def process_slices_length(new_slices, hm_slices):

    # new_slices, hm_slices
    difference = len(new_slices) - hm_slices

    if difference > 0:
        nth_slice = difference - 1
        for _ in range(difference):
            # if we have extra take last and 2nd last value average them and put it in 2nd last place
            # using mean for now but can have other operation in future
            new_val = list(map(mean, zip(*[new_slices[hm_slices+nth_slice],new_slices[hm_slices+(nth_slice-1)],])))
            # delete the last value
            del new_slices[hm_slices+nth_slice]
            # replace the 2nd last value with calculated new_val
            new_slices[hm_slices+(nth_slice-1)] = new_val
            nth_slice -= 1
    elif difference < 0:
        # if we have shortage slices then append the last slice back again
        for _ in range(abs(difference)):
            new_slices.append(new_slices[-1])
    else:
        pass
    return new_slices


def process_data(patient, labels_df, img_px_size=50, hm_slices=20, visualize=False):
    '''
    Note: Figure out the effect of changing the sequence ie. 1st processing the slices (fixed number) then resizing each slice.
    '''
    # patient, labels_df, img_px_size, hm_slices, visualize = patient, labels, img_size_px, slice_count, False
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]

    chunk_sizes = math.ceil(len(slices) / hm_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    new_slices = process_slices_length(new_slices, hm_slices)

    if visualize:
        fig = plt.figure()
        for num,each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            y.imshow(each_slice, cmap='gray')
        plt.show()

    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])

    return np.array(new_slices),label


def save_process_data():

    patients = os.listdir(data_dir)
    labels = pd.read_csv(lable_csv, index_col=0)

    much_data = []
    for num,patient in enumerate(patients):
        # num, patient = 6, patients[6]
        if num % 100 == 0:
            print(num)
        try:
            img_data,label = process_data(patient, labels, img_size_px, slice_count)
            # print(img_data.shape,label)
            much_data.append([img_data,label])
        except KeyError as e:
            print('This is unlabeled data!')

    np.save('./muchdata-{0}-{1}-{2}.npy'.format(img_size_px, img_size_px, slice_count), much_data)

    return 0


############ ConvNet Starts ############

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([54080,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


def train_neural_network(x, train_data, validation_data):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

    hm_epochs = 1
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            for data in train_data:
                total_runs += 1
                try:
                    epoch_loss = 0
                    X = data[0]
                    Y = data[1]
                    #print(Y)
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_run += 1
                except Exception as e:
                    pass
                    #print(str(e))

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('fitment percent:',successful_runs/total_runs)



def main():

    data_handling()

    save_process_data()

    much_data = np.load('./muchdata-{0}-{1}-{2}.npy'.format(img_size_px, img_size_px, slice_count)) #
    # If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
    train_data = much_data[:-2]
    validation_data = much_data[-2:]

    train_neural_network(x, train_data, validation_data)

    return 0



if __name__ == '__main__':
    main()
