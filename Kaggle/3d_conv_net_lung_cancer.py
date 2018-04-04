import pydicom as dicom # for reading dicom files
import os # for doing directory operations
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
# Change this to wherever you are storing your data:
# IF YOU ARE FOLLOWING ON KAGGLE, YOU CAN ONLY PLAY WITH THE SAMPLE DATA, WHICH IS MUCH SMALLER

# os.getcwd()
data_dir = './Kaggle/sample_images/'
lable_csv = './Kaggle/stage1_labels.csv'

img_size_px = 128
slice_count = 20


def chunks(l, n):
    '''
    Yield successive n-sized chunks from l.
    Get n chuncks from list l.
    '''
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


def process_data(patient,labels_df,img_px_size=50, hm_slices=20, visualize=False):
    '''
    Note: Figure out the effect of changing the sequence ie. 1st processing the slices (fixed number) then resizing each slice.
    '''
    # patient = '00cba091fa4ad62cc3200a657aeb957e'
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

    '''
    write a method for the below slice adjustment process
    '''
    # if we have shortage of one slice
    # append the last slice again
    if len(new_slices) == hm_slices-1:
        new_slices.append(new_slices[-1])

    # if we have shortage of two slice
    # append the last slice twice
    if len(new_slices) == hm_slices-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    # if we have excess of two slice
    # append the last slice twice
    if len(new_slices) == hm_slices+2:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val

    if len(new_slices) == hm_slices+1:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val

    if visualize:
        fig = plt.figure()
        for num,each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            y.imshow(each_slice, cmap='gray')
        plt.show()

    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])

    return np.array(new_slices),label








def main():

    data_handling()

    return 0



if __name__ == '__main__':
    main()
