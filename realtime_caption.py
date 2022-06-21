import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from pickle import load
from datetime import datetime

cap = cv2.VideoCapture(0)
ret,frame = cap.read()

# read trained word to index dictionary
wordtoix = load(open('./wordtoix.pkl','rb'))
ixtoword = load(open('./ixtoword.pkl','rb'))

# load trained model
model = tf.keras.models.load_model('./model_flickr8k')
#load feature extractor
#inception_v3 = InceptionV3(weights = 'imagenet')
#feature_extractor = Model(inception_v3.input,inception_v3.layers[-2].output)
feature_extractor = tf.keras.models.load_model('./feature_extractor')
max_length = 31

# font
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
fontColor = (0,255,255)
thickness = 1
lineType = 2

# load image and proprocess into Inception v3
def preprocess(frame):
	# resize it (width,heigth)
    x = cv2.resize(frame,(299,299),interpolation = cv2.INTER_AREA )
    x = np.expand_dims(x,axis =0) # (n_sample,width,height,colors)
    # preprocess_input
    x = preprocess_input(x) # scale it from 0-255 to 0-1 
    return x

# image embedding with feature_extractor
def encode(frame,feature_extractor):
    image = preprocess(frame) # convert image model input
    feature_vector = feature_extractor.predict(image) # get the encoding vector for the image
    # feature_vector = np.ravel(feature_vector) # reshape (1,2048) to (2048,)
    return feature_vector

# predict a caption
def greedySearch(feature_vector,verbose = 0):
  in_text ='startseq'
  for i in range(max_length):
    sequence = [wordtoix[word] for word in in_text.split() if word in wordtoix]
    sequence = pad_sequences([sequence],maxlen = max_length)
    yhat = model.predict([feature_vector,sequence],verbose = verbose) # [(1,2048),(1,31)]
    yhat = np.argmax(yhat) # (1610,)
    word = ixtoword[yhat] 
    in_text += ' ' + word
    if word == 'endseq':
      break
    final = in_text.split()
    final = final[1:-1] # list
    final = ' '.join(final) # convert list to string
  return final


if __name__ == "__main__":
	
	while ret:
		# get new frame
		ret,frame = cap.read()
		# Getting the current date and time
		dt = datetime.now()
		# extract feature
		feature_vector = encode(frame,feature_extractor)
		# predict caption
		caption = greedySearch(feature_vector,0)
		# put text
		cv2.putText(frame,str(dt),(5,20),font,fontScale,fontColor,thickness,lineType)
		cv2.putText(frame,caption,(5,40),font,fontScale,fontColor,thickness,lineType)

		cv2.imshow('frame',frame)

		if cv2.waitKey(1) == 27:
			break

	cv2.destroyAllWindows()
	cap.release()