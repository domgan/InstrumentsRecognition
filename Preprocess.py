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


P = part_train('piano', 31)
G = part_train('guitar', 31)
D = part_train('drums', 31)
train_input = np.concatenate((P, G, D))


train_labels = np.zeros(93)
for i in range(31,62):
    train_labels[i] = 1
for i in range(62, 93):
    train_labels[i] = 2


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


test_labels = np.zeros(33)
for i in range(11,22):
    test_labels[i] = 1
for i in range(22, 33):
    test_labels[i] = 2
