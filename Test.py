import os
print('Loading Tensorflow...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disable console info from tensorflow

import Preprocess
from tensorflow import keras
import numpy as np

model = keras.models.load_model('model/model.h5')


def instrument(path):
    p = Preprocess.melspec(path)

    p = np.expand_dims(p, 0)
    p = np.expand_dims(p, 3)

    predictions_single = model.predict(p)
    print(predictions_single[0])
    predictions = predictions_single[0]
    pins = ''
    gins = ''
    dins = ''
    if predictions[0] > 0.5:
        pins = 'Piano '
    elif predictions[1] > 0.5:
        gins = 'Guitar '
    elif predictions[2] > 0.5:
        dins = 'Drums '
    else:
        print('----')
    print(pins + gins + dins)


instrument('data/predict/pp0.wav')
instrument('data/predict/pg0.wav')
instrument('data/predict/pd0.wav')
