
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import matplotlib.pyplot as plt


# os.getcwd()
train_dir = './data/train/'
test_dir = './data/test/'
img_size = 128
learning_rate = 1e-3
model_name = 'dogsvscats-{}-{}.model'.format(learning_rate, '6conv-basic') # just so we remember which saved model is which, sizes must match

save_training_data = 1
save_testing_data = 1
save_model = 1


def label_img(img_name):
    '''
    Split the image name to compute class of the image.
    '''
    # img_name = img
    word_label = img_name.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    if word_label == 'cat':
        return [1,0] # [much cat, no dog]
    elif word_label == 'dog':
        return [0,1] # [no cat, very doggo]


def create_train_data():
    '''
    Create list of train data consisting grayscale features of image in an array and respective label.
    '''

    if os.path.exists('./train_data.npy'):
        print('Loading saved train data')
        training_data = np.load('./train_data.npy')
        return training_data
    else:
        print('No saved train data')

    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        # img = os.listdir(train_dir)[0]
        label = label_img(img)
        path = os.path.join(train_dir,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        # array of list representing pixel values in rows
        img = cv2.resize(img, (img_size,img_size))
        # resize images to img_size x img_size
        training_data.append([np.array(img),np.array(label)])
        # creating lists of data consisting feature array and respective labes
    shuffle(training_data)

    if save_training_data:
        np.save('./train_data.npy', training_data)

    return training_data


def process_test_data():
    '''
    Create list of test data consisting grayscale features of image in an array.
    '''

    if os.path.exists('./test_data.npy'):
        print('Loading saved test data')
        training_data = np.load('./test_data.npy')
        return training_data
    else:
        print('No saved test data')

    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)

    if save_testing_data:
        np.save('./test_data.npy', testing_data)

    return testing_data


def load_model(convnet):

    model = tflearn.DNN(convnet)

    if os.path.exists('./{}.meta'.format(model_name)):
        model.load('./'+model_name)
        print('model loaded!')
    else:
        print('Empty model loaded!')
    return model

def get_convnet():

    convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')

    return convnet

def train_model(train_data):

    convnet = get_convnet()

    model = load_model(convnet)

    # train_data = create_train_data()

    train = train_data[:-500]
    test = train_data[-500:]

    X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}),
                                            snapshot_step=500, show_metric=True, run_id=model_name)

    if save_model:
        model.save('./'+model_name)

    return model


def predict_from_model(model, data, label=False):

    model_out = model.predict([data])[0]

    str_label = ''
    if np.argmax(model_out) == 1:
        str_label='Dog'
    else:
        str_label='Cat'

    if label:
        return str_label

    return model_out


def predict_test_data(model, test_data):

    fig=plt.figure()

    for num,data in enumerate(test_data[:12]):
        # cat: [1,0]
        # dog: [0,1]

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(img_size,img_size,1)
        #model_out = model.predict([data])[0]
        str_label = predict_from_model(model, data, True)

        y.imshow(orig,cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()

    return 0

def submit_prediction(model, test_data):

    with open('./submission_file.csv','a') as f:
        f.write('id,label\n')
        for data in tqdm(test_data):
            img_num = data[1]
            img_data = data[0]
            orig = img_data
            data = img_data.reshape(img_size,img_size,1)
            # model_out = model.predict([data])[0]
            model_out = predict_from_model(model, data)
            f.write('{},{}\n'.format(img_num,model_out[1]))

    return 0


def main():

    train_data = create_train_data()
    model = train_model(train_data)

    test_data = process_test_data()

    predict_test_data(model, test_data)

    submit_prediction(model, test_data)

    return 0



if __name__ == '__main__':
    main()
