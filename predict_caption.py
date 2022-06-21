import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.inception_v3 import preprocess_input
from pickle import load

# init parser
# parser = argparse.ArgumentParser(description = 'Use trained model t')