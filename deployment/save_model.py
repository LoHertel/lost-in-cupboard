import argparse
import os

# supress info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
 

# parse arguments
parser = argparse.ArgumentParser(description="Saves Keras HDF5 model in SavedModel format on disk.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model", help="Source location of model with .h5 file type")
parser.add_argument("name", help="Target name for the SavedModel folder")
args = parser.parse_args()
config = vars(args)


# load model, which was trained with Keras
model = keras.models.load_model(config['model'])

# export into SaveModel format
tf.saved_model.save(model, config['name'])

print(f"Save model into {config['name']} finished.")