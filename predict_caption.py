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
import matplotlib.pyplot as plt

# init parser
parser = argparse.ArgumentParser(description = 'Use trained model to predict caption of image')
# add argument to parser
parser.add_argument('-i','--img',type = str, help = 'directory of image',required = True)
parser.add_argument('-s','--show',action = 'store_true',help = 'option to display the image')
# create arguments
args = parser.parse_args()

# read trained word to index dictionary
wordtoix = load(open('./wordtoix.pkl','rb'))

# load trained model
model = tf.keras.models.load_model('./model_flickr8k')
#load feature extractor
#inception_v3 = InceptionV3(weights = 'imagenet')
#feature_extractor = Model(inception_v3.input,inception_v3.layers[-2].output)
feature_extractor = tf.keras.model.load_model('./feature_extractor')

# load image and proprocess into Inception v3
def preprocess(image_path):
    # convert all the images to size 299x299 as excepted by the inception v3 model
    img = image.load_img(image_path,target_size=  (299,299))
    # convert image into array
    x= image.img_to_array(img)
    # add 1 dimension as 1 simple
    x = np.expand_dims(x,axis =0) # (n_sample,width,height,colors)
    # preprocess_input
    x = preprocess_input(x) # scale it from 0-255 to 0-1 
    return x

# image embedding with feature_extractor
def encode(image,feature_extractor):
    image = preprocess(image) # convert image model input
    feature_vector = feature_extractor.predict(image) # get the encoding vector for the image
    # feature_vector = np.ravel(feature_vector) # reshape (1,2048) to (2048,)
    return feature_vector

# predict a caption
def greedySearch(feature_vector,verbose = 0):
  in_text ='startseq'
  for i in range(max_length):
    sequence = [wordtoix[word] for word in in_text.split() if word in wordtoix]
    sequence = pad_sequences([sequence],maxlen = max_length)
    yhat = model_word.predict([feature_vector,sequence],verbose = verbose) # [(1,2048),(1,31)]
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
	feature_vector = encode(args.img,feature_extractor) # (2048,)
	img = plt.imread(args.img)
	plt.imshow(img)
	plt.title(greedySearch(feature_vector,0))
	plt.show()