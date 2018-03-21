import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd

lemmatizer = WordNetLemmatizer()

# Sentiment140 dataset
# Download the training and test data zip file from pythonprograming.net or link in the README.md file

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

def init_process(fin,fout):
	'''
	This method takes a data like: "0","1467810369","Mon Apr 06 22:19:45 PDT 2009","NO_QUERY","_TheSpecialOne_","@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"
	And convert it into data: [1, 0]::: that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D
	Above is and example of a negative statement. represented by "0" in the begining
	And thus the train/test_set.csv consists [1,0] lable for negative and [0,1] lable for positive
	Although dataset consists neutral sentences, we are ignoring them for now.
	'''
	# fin,fout = './training.1600000.processed.noemoticon.csv','./train_set.csv'
	outfile = open(fout,'a')
	with open(fin, buffering=200000, encoding='latin-1') as f:
		# print(f)
		# putting try catch inside the loop to continue with remaining line even if an exception
		for line in f:
			try:
				line = line.replace('"','')
				initial_polarity = line.split(',')[0]
				# polarity 0 : a Positive sentence
				if initial_polarity == '0':
					initial_polarity = [1,0]
				# polarity 4 : a Neagative sentence
				elif initial_polarity == '4':
					initial_polarity = [0,1]
				else:
					# This is to avoide polarity 2 line adding in csv file
					print("Polarity not processed:", line)
					continue
				tweet = line.split(',')[-1]
				outline = str(initial_polarity)+':::'+tweet
				outfile.write(outline)
			except Exception as e:
				print(str(e))
				continue;
	outfile.close()

init_process('./training.1600000.processed.noemoticon.csv','./train_set.csv')
# creates a "train_set.csv" consisting positive, negative sentences with respective lables
init_process('./testdata.manual.2009.06.14.csv','./test_set.csv')
# creates a "test_set.csv" consisting positive, negative sentences with respective lables

def create_lexicon(fin):
	# fin = "./train_set.csv"
	lexicon = []
	with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
		try:
			counter = 1
			content = ''
			for line in f:
				counter += 1
				if (counter/2500.0).is_integer():
					# lexiconing from line 2500
					tweet = line.split(':::')[1]
					content += ' '+tweet
					words = word_tokenize(content)
					words = [lemmatizer.lemmatize(i) for i in words]
					lexicon = list(set(lexicon + words))
					print(counter, len(lexicon)) # 1594577 2569

		except Exception as e:
			print(str(e))

	with open('lexicon-2500-2638.pickle','wb') as f:
		pickle.dump(lexicon,f)

create_lexicon('./train_set.csv')
# creates a ".pickle" file consisting all the words lexiconed as a list


def convert_to_vec(fin,fout,lexicon_pickle):
	'''
	This method is similar to "sample_handling" in "create_sentiment_featuresets.py"
	'''
	# fin,fout,lexicon_pickle = './test_set.csv','./processed-test-set.csv','./lexicon-2500-2638.pickle'
	with open(lexicon_pickle,'rb') as f:
		lexicon = pickle.load(f)
	outfile = open(fout,'a')
	with open(fin, buffering=20000, encoding='latin-1') as f:
		# f=f # to single line run the code
		counter = 0
		for line in f:
			counter +=1
			label = line.split(':::')[0] # split and take the lable
			tweet = line.split(':::')[1] # split and take the text
			# label, tweet = line.split(':::') # above split done in single line

			current_words = word_tokenize(tweet.lower())
			# get every word in a list from the line text
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			# lemmatize every word in the list

			features = np.zeros(len(lexicon))
			# create an array of features of the size of lexicon

			for word in current_words:
				# word = current_words[0]
				if word.lower() in lexicon:
					# if a word exist in the list of lexicon get its index value
					index_value = lexicon.index(word.lower())
					# OR DO +=1, test both
					features[index_value] += 1
					# increment the feature value at index of the word in lexicon

			features = list(features)
			outline = str(features)+'::'+str(label)+'\n'
			# feature array with label
			outfile.write(outline)

		print(counter)

convert_to_vec('./test_set.csv','./processed-test-set.csv','./lexicon-2500-2638.pickle')
# create "processed-test-set.csv" consisting feature array of each text with respective lables
# Note: few features with label of polarity 2 where processed in "processed-test-set.csv"
# This was due to "test_set.csv" consisting polarity 2, Handled in "init_process()"

def shuffle_data(fin):
	'''
	This method reads the csv file to dataframe randomly shuffels the data and write it back to a csv file
	'''
	# fin = './train_set.csv'
	df = pd.read_csv(fin, error_bad_lines=False, encoding ='latin-1')
	# encoding ='latin-1' solves the previous read_csv() errors
	df = df.iloc[np.random.permutation(len(df))]
	print(df.head())
	df.to_csv('./train_set_shuffled.csv', index=False)

shuffle_data('./train_set.csv')


def create_test_data_pickle(fin):
	'''
	For now this method reads the csv files consisting array of features and respective lables
	and splits them into seperate list of "feature_sets", "labels"
	'''
	# fin = './processed-test-set.csv'
	feature_sets = []
	labels = []
	counter = 0
	with open(fin, buffering=20000) as f:
		for line in f:
			try:
				features = list(eval(line.split('::')[0]))
				label = list(eval(line.split('::')[1]))

				feature_sets.append(features)
				labels.append(label)
				counter += 1
			except:
				pass
	print(counter)
	feature_sets = np.array(feature_sets)
	labels = np.array(labels)

create_test_data_pickle('./processed-test-set.csv')
