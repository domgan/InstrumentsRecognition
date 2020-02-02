import numpy as np
import librosa
import scipy


def melspec(path):  # load wave and compute melspectrogram
    bits = 16
    #fs, y = scipy.io.wavfile.read(path)
    #y = np.float64(y)
    #y = y.T
    #y = librosa.core.to_mono(y)
    y, fs = librosa.core.load(path)
    y = y[0:150000]
    if np.max(y) > 1:
        y = y / (2 ** (bits - 1))
    data = librosa.feature.melspectrogram(y, fs)
    return data


""" Train Input """


def part_train(ins, num_of_samples):
    arr = []
    if ins == 'piano':
        for i in range(num_of_samples):
            data = melspec('data/p'+str(i)+'.wav')
            arr.append(data)
        return np.array(arr)
    elif ins == 'guitar':
        for i in range(num_of_samples):
            data = melspec('data/g'+str(i)+'.wav')
            arr.append(data)
        return np.array(arr)
    elif ins == 'drums':
        for i in range(num_of_samples):
            data = melspec('data/d'+str(i)+'.wav')
            arr.append(data)
        return np.array(arr)


P = part_train('piano', 41)
G = part_train('guitar', 31)
D = part_train('drums', 31)
train_input = np.concatenate((P, G, D))


Pt = np.zeros((41,3))
Pt[:, 0] = 1
Gt = np.zeros((31,3))
Gt[:, 1] = 1
Dt = np.zeros((31,3))
Dt[:, 2] = 1
train_labels = np.concatenate((Pt, Gt, Dt))


""" Test Input """


def part_test(ins, num_of_samples):
    arr = []
    if ins == 'piano':
        for i in range(num_of_samples):
            data = melspec('data/test/tp'+str(i)+'.wav')
            arr.append(data)
        return np.array(arr)
    elif ins == 'guitar':
        for i in range(num_of_samples):
            data = melspec('data/test/tg'+str(i)+'.wav')
            arr.append(data)
        return np.array(arr)
    elif ins == 'drums':
        for i in range(num_of_samples):
            data = melspec('data/test/td'+str(i)+'.wav')
            arr.append(data)
        return np.array(arr)


TP = part_test('piano', 11)
TG = part_test('guitar', 11)
TD = part_test('drums', 11)
test_input = np.concatenate((TP, TG, TD))


TPt = np.zeros((11, 3))
TPt[:, 0] = 1
TGt = np.zeros((11, 3))
TGt[:, 1] = 1
TDt = np.zeros((11, 3))
TDt[:, 2] = 1
test_labels = np.concatenate((TPt, TGt, TDt))


"""adding fourth dimension for cNN"""

train_input = np.expand_dims(train_input, 3)
test_input = np.expand_dims(test_input, 3)
