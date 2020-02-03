import numpy as np
import librosa
import scipy


def melspec(path):  # load wave and compute melspectrogram
    bits = 16
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
    elif ins == 'combined':
        for i in range(num_of_samples):
            data = melspec('data/c'+str(i)+'.wav')
            arr.append(data)
        return np.array(arr)


num_of_P = 41
num_of_G = 31
num_of_D = 31
num_of_C = 2


P = part_train('piano', num_of_P)
G = part_train('guitar', num_of_G)
D = part_train('drums', num_of_D)
C = part_train('combined', num_of_C)
train_input = np.concatenate((P, G, D, C))


Pt = np.zeros((num_of_P, 3))
Pt[:, 0] = 1
Gt = np.zeros((num_of_G, 3))
Gt[:, 1] = 1
Dt = np.zeros((num_of_D, 3))
Dt[:, 2] = 1
Ct = np.zeros((num_of_C, 3))
Ct[:, 0] = 1
Ct[:, 2] = 1
train_labels = np.concatenate((Pt, Gt, Dt, Ct))


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
