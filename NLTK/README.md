
I started with data_preprocessing.py file with "Sentiment140 dataset"
Use this link to download the zip file of data: http://help.sentiment140.com/for-students/

running  method "init_process()" with training and test csv file to create train and test dataset

I made few changes in "init_process()" for ignoring the polarity 2 line being processed  
from "./testdata.manual.2009.06.14.csv" to "./test_set.csv"

In method "shuffle_data()" use " encoding ='latin-1' " if you get utf-8 errors while read_csv() in pandas dataframe
