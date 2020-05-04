'''

SRL.py
AIT 726 Assignment 4: Semantic Role Labeling
Haritha G, Giridhar K

Spring 2020
Instructions for execution

Pseudocode
1. Read training data
	Repeat data point with one unique verb/predicate if 
		the number of verbs is >1
		Represent the verb under consideration by a flag
2. Embedding
	Using GloVe embeddings 
		50 dimensions
3. Build a Feed Forward Neural Network
	3 Hidden layers of size 20 each


'''

import os
import sys
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import timeit
import csv
import bisect

from collections import defaultdict
from pathlib import Path

import nltk as nl
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


#from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

'''
This method reads and returns the data given the data path
1. reads data from given path
2. Reframes the data such that the sentence is repeated per number of verbs
3. returns the dataframe
'''
def get_data(path,dataname):

	newlist=[]
	#dataset= pd.read_csv(path,sep="\t",header= None)

	data= open(path,"r")
	dataset= data.readlines()
	for d in dataset:
		newlist.append(d.split())

	newlist.append([]) 
	maxlen= max(len(x) for x in newlist)
	print("sentence of max length is ",maxlen)

	otherlist=[]
	for l in newlist:
		if len(l)< maxlen:
			l= l+([0]*(maxlen-len(l)))
			otherlist.append(l)

	# print(newlist)


	df= pd.DataFrame(otherlist)
	names= ["words","synt_col2_pos","synt_col2_fp","synt_col2h_pos","synt_col2h_fp","synt_upc_pos","synt_upc_pp","synt_cha_pos","synt_cha_fp","senses","ne", "props","SRL_1","SRL_2","SRL_3","SRL_4","SRL_5","SRL_6","SRL_7","SRL_8"]
	exceptverbs= ["words","synt_col2_pos","synt_col2_fp","synt_col2h_pos","synt_col2h_fp","synt_upc_pos","synt_upc_pp","synt_cha_pos","synt_cha_fp","senses","ne"]
	names2= ["words","synt_col2_pos","synt_col2_fp","synt_col2h_pos","synt_col2h_fp","synt_upc_pos","synt_upc_pp","synt_cha_pos","synt_cha_fp","senses","ne", "props","SRL_1","SRL_2","SRL_3","SRL_4","SRL_5","SRL_6","SRL_7"]
	if dataname== "train":
		df.columns =  names
	elif dataname=="test":
		df.columns= names2

	#print(df['props'])

	df.to_csv("traindata.csv")

	#df=df.loc[:100,:]

	#print(df.head(15))
	d1= pd.DataFrame()
	sent=[]
	previous=0
	indices= df[df.words == 0].index

	for i in indices:
		#dataframe of one sentence
		d1= df.loc[previous:i-1]
		

		noofverbs= len(d1[d1['props'] != '-'])
		if noofverbs>1:
			otherd1= d1[exceptverbs]
			verbs= d1["props"].values
			#print(verbs)
			verbindices = [l for l, x in enumerate(verbs) if x !="-"]
			#print(verbindices)
			for j in range(0,noofverbs):
				verbs2= verbs.copy()
				d2= otherd1.copy()

				for k1, k2 in enumerate(verbindices):
					#print(k1,k2)
					if(k1!=j):
						# print("printing verb2")
						# print(verbs2[k2])
						verbs2[k2]= "-"
				d2["props"]= verbs2
				d2["SRL_1"]= d1.iloc[:,[len(exceptverbs)+j+1]]
				# print(d2)
				sent.append(d2)
		else:
			#print(d1)
			sent.append(d1[exceptverbs+["props","SRL_1"]])
		

		previous=i+1

	BigDF= pd.concat(sent,sort=False)
	print(BigDF.shape)

	return BigDF

'''
This method takes care of splitting the data
1.randomize data points
2. split ratio 80:20 for train and validation
'''
def train_validation_set(df,y):
	#train, validation and test
	X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=1, shuffle=False)


	print("shapes of train validation sets" )
	print(X_train.shape)
	print(X_val.shape)

	return X_train, X_val, y_train, y_val

'''
This method reads loads the glove embedding vector representation
'''
def glove_embedding():
	embeddings_dict= {}
	with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
		for line in f:
			values = line.split()
			word = values[0]
			vector = np.asarray(values[1:], "float32")
			embeddings_dict[word] = vector
	return embeddings_dict

'''
This method creates vocab for unique words in the corpus
'''
def createvocab(text):
	  V=[]
	  for word in text:
		  if word in V:
			  continue
		  else :
			  V.append(word)
	  return V

'''
This method takes text and embeddings as input
1.look up embeddings for the words from GLoVe
2. retuns the embeddings that would act as weights later on for neural network
'''
#look up embeddings for the words from GLoVe
def createWeightsLookup(text, embedding,embedding_dim):
  vocab = createvocab(text)
  print('vocab length is ', len(vocab))

  weights_matrix = np.zeros((len(text), embedding_dim))
  words_found = 0

  #Create dictionary of words and representation from pretrained embeddings
  for i, word in enumerate(text.values):
	  try: 
		  word = str(word)
		  weights_matrix[i] = embedding[word]
		  words_found += 1
	  except KeyError:
		  weights_matrix[i] = np.zeros((embedding_dim,))

  print('Created weights matrix from pre trained that would act as a basis for lookup table')
  return vocab, weights_matrix


'''
Feedforward neural network class
3 layers, 2 hidden and one output layer
input size : feature set size
hidden size : 20
output size : labelsize
hidden activation : sigmoid
output activaltion : softmax
'''

class FFNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(FFNN, self).__init__() #name should be the same as the classname
		self.input_size= input_size
		self.hidden_size= hidden_size
		self.fc1 = nn.Linear(self.input_size,self.hidden_size)
		self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
		self.fc3= nn.Linear(self.hidden_size, 60)
		self.sigmoid = nn.Sigmoid()
		self.softmax= nn.Softmax()

	def forward(self, x):
		a1 = self.sigmoid(self.fc1(x))
		a2= self.sigmoid(self.fc2(a1))
		a3= self.fc3(a2)
		a4= self.softmax(a3)
		return a4

'''
training runs for 
100 epochs
1000 batch size
0.01 learning rate.
For every epoch:
	The training is done in batches - forward pass and back propagate
	Validation is performed and accuracy and loss is computed

train loss, validation loss and validatiuon accuracy are plotted

'''
def training(X_train, y_train, X_val, y_val,lr= 0.01,epochs=100,batch_size=1000):
	model= FFNN(X_train.shape[1], 20)
	trainloss = []
	testloss = []
	testaccuracy = []
	print("\t Model Summary:")
	print("\t ", model)
	trainloss = []
	starttime = time.time()
	for e in range(epochs):
		loss_fn= F.cross_entropy
		optimizer= optim.SGD(model.parameters(), lr=lr)

		#converting data to tensors
		#y_train= torch.from_numpy(y_train.reshape((y_train.shape[0],-1))).float()
		X_train= torch.FloatTensor(X_train)
		y_train= torch.FloatTensor(y_train)
	
		X_val= torch.FloatTensor(X_val)
		y_val= torch.FloatTensor(y_val)
		
		train_tsdata= torch.utils.data.TensorDataset(X_train,y_train)
		val_tsdata= torch.utils.data.TensorDataset(X_val,y_val)

		#data loader divides data into batches
		train_loader= torch.utils.data.DataLoader(train_tsdata, batch_size= batch_size, shuffle= True)
		valid_loader= torch.utils.data.DataLoader(val_tsdata, batch_size= batch_size, shuffle= False)

		#setting model to train mode: This means the weights will be updated
		model.train()

		avg_loss = 0.
		for x_batch, y_batch in train_loader:
			# Variable is the pytorch wrapper for tensors
			X = Variable(torch.FloatTensor(x_batch))
			y = Variable(torch.LongTensor(y_batch.long()))
			#null the gradients
			optimizer.zero_grad()
			#forward pass
			y_pred = model(X)
			  #Compute loss
			loss = loss_fn(y_pred, y)
			
			#back propagate
			loss.backward()
			optimizer.step()
			
			avg_loss += loss.item() / len(train_loader)
		trainloss.append(avg_loss)
		model.eval()
		avg_val_loss = 0.
		testacc = 0
		for x_batch, y_batch in valid_loader:
			X_l = Variable(torch.FloatTensor(x_batch))
			y_l = Variable(torch.LongTensor(y_batch.long()))
			y_pred = model(X_l.float()) 

			avg_val_loss += loss_fn(y_pred, y_l).item() / len(valid_loader)
			predictions= y_pred.argmax(dim=1).numpy()
			labels = y_batch.float().numpy() #Truth
			testacc += np.sum(predictions == labels) 
		print('\t Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(e + 1, epochs, avg_loss, avg_val_loss, time.time()-starttime))
		testloss.append(avg_val_loss)
		testaccuracy.append(testacc/ len(y_val))

	plt.title("plot of train,val loss and val accuracy for lr = {}".format(lr))
	plt.plot(trainloss)
	plt.plot(testloss)
	plt.show()
	plt.title("plot of val accuracy for lr = {}".format(lr))
	plt.plot(testaccuracy)
	plt.show()
	return model



'''
This methods return the predictions for test data
'''

def testing(model, test_data, test_labels):

	test_X = Variable(torch.from_numpy(test_data)).float()

	#set to evaluation mode
	model.eval()

	y_pred = model(test_X.float()) #Get predictions
	# predictions = np.round(y_pred[:, 0].data.numpy())

	predictions= y_pred.argmax(dim=1).numpy()
	#compute accuracy and confusion matrix
	cm = confusion_matrix(test_labels, predictions, labels=None)
	print("\t Confusion matrix \n \t {}".format(cm))
	print("\t Accuracy is {}".format(np.round(np.sum(predictions == test_labels) / len(test_X), 2)))
	return y_pred,predictions


def main():

	#writing to log file
	stdoutorigin = sys.stdout 
	sys.stdout = open("/Users/gkbytes/SRL/Status2.txt", "w")


	#get data
	dataset= get_data("/Users/gkbytes/SRL/training Data/training7.txt","train")
	test_data= get_data("/Users/gkbytes/SRL/training Data/testing.txt","test")
	
	dataset['SRL_1']= dataset.SRL_1.astype(str)
	test_data['SRL_1']= test_data.SRL_1.astype(str)

	y= dataset['SRL_1']
	y_test= test_data['SRL_1']

	#adding a flag for the verb
	dataset['verb_status']= dataset.apply(lambda row: row.words== row.props,axis=1)
	test_data['verb_status']=test_data.apply(lambda row: row.words== row.props,axis=1)
	
	dataset= dataset.drop(columns= 'SRL_1')


	#make train and validate sets
	X_train,X_val, y_train, y_val = train_validation_set(dataset, y)
	columns_to_include = ["synt_col2_pos","synt_col2h_pos","synt_upc_pos","synt_cha_fp","verb_status"]
	
	#embedding dictionary
	embeddings=glove_embedding()

	# create embedding matrix
	vocab,embedding_matrix=createWeightsLookup(X_train['words'], embeddings,50)
	valid_vocab,validation_embedding_matrix = createWeightsLookup(X_val['words'], embeddings, 50)
	test_vocab, test_embedding_matrix= createWeightsLookup(test_data['words'], embeddings, 50)
	
	print(type(embedding_matrix))
	print(embedding_matrix.shape)

	#one hot encode the input columns
	ohe= preprocessing.OneHotEncoder(handle_unknown='ignore')
	allother_input= ohe.fit(dataset[columns_to_include])
	other_input =  ohe.transform(X_train[columns_to_include])
	other_input_validation= ohe.transform(X_val[columns_to_include])
	other_input_test=ohe.transform(test_data[columns_to_include])

	print(type(other_input))
	print(other_input.shape)
	#Concatenate word and other columns into a dataframe

	full_train_matrix= np.concatenate((embedding_matrix,other_input.toarray()), axis=1)
	full_valid_matrix= np.concatenate((validation_embedding_matrix,other_input_validation.toarray()), axis=1)
	full_test_matrix=  np.concatenate((test_embedding_matrix,other_input_test.toarray()), axis=1)
	print(len(np.unique(y_train)))

	#Label encode the output feature
	le = preprocessing.LabelEncoder()
	
	y = le.fit(y.values)
	y_train = le.transform(y_train.values)
	y_val= le.transform(y_val.values)
	
	y_test = y_test.map(lambda s: 'other' if s not in le.classes_ else s)
	le_classes = le.classes_.tolist()
	bisect.insort_left(le_classes, 'other')
	le.classes_ = le_classes
	y_test= le.transform(y_test)

	print("Label encoder classes-",le.classes_)


	#training
	final_model=training(full_train_matrix,y_train, full_valid_matrix, y_val)
	torch.save(final_model.state_dict(),"/Users/gkbytes/SRL/FFNN_SRL.pt")
	
	#use final model for prediction on test data
	test_y_pred,test_labels= testing(final_model, full_test_matrix, y_test)


	#dataframe with ground truth, predictions and labels
	d = pd.DataFrame(list(zip(y_test,test_labels,test_y_pred)), columns = ["ground truth","test labels","test preds"])

	



if __name__ == "__main__":
	main()
