import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
import os
from collections import Counter
from nltk.stem import WordNetLemmatizer

save_train_test_data = 0

lemmatizer = WordNetLemmatizer()
# how many lines from file
hm_lines = 100000
# tokenize only 100000 line if file consists more than 100000 line

def create_lexicon(pos,neg):
	# pos,neg = './pos.txt','./neg.txt'

	lexicon = []
	# lexicon is a list consisting all the words from file './pos.txt','./neg.txt'
	with open(pos,'r') as f:
		contents = f.readlines()
		# len(contents) # lines in file
		# len(contents[:hm_lines])
		for l in contents[:hm_lines]:
			# l = contents[0]
			all_words = word_tokenize(l)
			# all_words is a list consisting each word from the line l
			lexicon += list(all_words)
			# as all_words is a list above list() can be ignored

	with open(neg,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)

	# len(lexicon) # with just './pos.txt' = 115419, and just './neg.txt' 114821
 	# conbined its 230240

	# This is just to check lemmatizer.lemmatize() does on few words
	# temp_lexicon = [lexicon[i] for i in range(1000)]
	# temp_lexicon_lemmatize = [lemmatizer.lemmatize(i) for i in temp_lexicon]

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	# makes word singular from plural
	# eg: rings -> ring, movies -> movie
	w_counts = Counter(lexicon)
	# count occurance of each word
	# total words 230240 only 20326 unique
	l2 = []
	for w in w_counts:
		# w = w_counts['dramatic']
		if 1000 > w_counts[w] > 50:
			# get the words whose occurance is more than 50 and less than 1000
			# less than 1000 because we dont want common words
			l2.append(w)
	print(len(l2))
	return l2


def sample_handling(sample,lexicon,classification):
	# sample = 'pos.txt'
	# classification = [1,0]
	# here we are reading each line in txt file
	# making list of of size lexicon and incrementing the nth value in the list if the word exists in lexicon

	featureset = []

	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			# l = contents[0]
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				# word = current_words[0]
				if word.lower() in lexicon:
					# if a word exist in the list of lexicon get its index value
					index_value = lexicon.index(word.lower())
					# increment the feature value at index of the word in lexicon
					features[index_value] += 1

			features = list(features)
			featureset.append([features,classification])

	return featureset



def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
	# pos,neg = './pos.txt','./neg.txt'
	lexicon = create_lexicon(pos,neg)
	# lexicon = l2 # here lexicon list of words after "lemmatize" whose occurance are in range 51-999
	features = []
	features += sample_handling('pos.txt',lexicon,[1,0])
	# list of positive feature and there class label is added to this list with the help of 'pos.txt'
	features += sample_handling('neg.txt',lexicon,[0,1])
	# list of negative feature and there class label is added to this list with the help of 'neg.txt'
	random.shuffle(features)
	features = np.array(features)

	testing_size = int(test_size*len(features))
	# take testing_size as x % of features

	train_x = list(features[:,0][:-testing_size])
	# 100 - x % of features as training data
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	# x % of features as testing data
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y


if __name__ == '__main__':
	# os.path.isfile("./pos.txt"), os.path.isfile("./neg.txt")
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('./pos.txt','./neg.txt')
	# if you want to pickle this data:
	if save_train_test_data:
		with open('/path/to/sentiment_set.pickle','wb') as f:
			pickle.dump([train_x,train_y,test_x,test_y],f)
