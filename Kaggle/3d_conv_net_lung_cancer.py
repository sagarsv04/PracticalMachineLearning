import pydicom as dicom # for reading dicom files
import os # for doing directory operations
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import matplotlib.pyplot as plt

# Change this to wherever you are storing your data:
# IF YOU ARE FOLLOWING ON KAGGLE, YOU CAN ONLY PLAY WITH THE SAMPLE DATA, WHICH IS MUCH SMALLER

# os.getcwd()
data_dir = './Kaggle/sample_images/'
lable_csv = './Kaggle/stage1_labels.csv'
patients = os.listdir(data_dir)
labels_df = pd.read_csv(lable_csv, index_col=0)

labels_df.head()

for patient in patients[:1]:
    # patient = patients[:1][0]
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient

    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    print(len(slices),label)
    print(slices[0])


for patient in patients[:3]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient

    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    print(slices[0].pixel_array.shape, len(slices))


len(patients)


for patient in patients[:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    #          the first slice
    plt.imshow(slices[0].pixel_array)
    plt.show()
