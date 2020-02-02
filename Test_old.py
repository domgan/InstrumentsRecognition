import os
print('Loading Tensorflow...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable console info from tensorflow

import Preprocess_old as Preprocess
from tensorflow import keras
import numpy as np

model = keras.models.load_model('model.h5')

p = Preprocess.melspec('data/predict/pp0.wav')

p = np.expand_dims(p, 0)
p = np.expand_dims(p, 3)

predictions_single = model.predict(p)
print(predictions_single[0])

if np.argmax(predictions_single[0]) == 0:
    ins = 'Piano'
    print(ins)
elif np.argmax(predictions_single[0]) == 1:
    ins = 'Guitar'
    print(ins)
elif np.argmax(predictions_single[0]) == 2:
    ins = 'Drums'
    print(ins)
