import os
import sys
import time
import numpy as np
import scipy
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc
from keras.layers import TimeDistributed, Bidirectional, add, Input
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Activation, Conv2D, Conv1D, MaxPool1D, Flatten, Reshape, Dropout, \
    BatchNormalization, AveragePooling1D, MaxPool2D
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop, Adadelta
from keras import regularizers
from keras import backend as K


# attain the file path list of the dataset
def get_wav_files(wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)
    return wav_files


def make_feature(wav_files):
    train_x = []
    train_y = []
    paragraph_label = []
    begin_time = time.time()
    for i, onewav in enumerate(wav_files):

        if i % 5 == 4:
            gaptime = time.time() - begin_time
            percent = float(i) * 100 / len(wav_files)
            eta_time = gaptime * 100 / (percent + 0.01) - gaptime
            strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
            str_log = ("%.2f %% %s %s/%s \t used:%ds eta:%d s" % (
                percent, strprogress, i, len(train_y), gaptime, eta_time))
            sys.stdout.write('\r' + str_log)

        label = onewav.split("\\")[1].split('_')[0]

        (rate, sig) = wav.read(onewav)
        # print(rate)
        mfcc_feat = mfcc(sig, rate, winlen=0.025, winstep=0.01, nfft=512)
        # mfcc_feat = mfcc(scipy.signal.resample(sig, len(sig) // 2), rate // 2)

        d_mfcc_feat = delta(mfcc_feat, 2)
        dd_mfcc_feat = delta(d_mfcc_feat, 2)

        finalfeature = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
        # print(finalfeature.shape)
        train_x.append(finalfeature)
        train_y.append(label[1:])
        paragraph_label.append(label[0])

    # yy = LabelBinarizer().fit_transform(train_y)
    yy = binarizer(train_y)

    # define the input feature shape (1561, 39) for each wav.file, padded
    for i in range(len(train_x)):
        length = train_x[i].shape[0]
        if length < 1024:
            train_x[i] = np.concatenate((train_x[i], np.zeros((1024 - length, 39))), axis=0)
        else:
            train_x[i] = train_x[i][:1024, :]
            print('Oops exceeds,', train_x[i].shape)
    # train_x = [np.concatenate((j, np.zeros((1561-j.shape[0], 39)))) for j in train_x]
    train_x = np.asarray(train_x)
    train_y = np.asarray(yy)
    paragraph_label = np.asarray(paragraph_label)
    print(paragraph_label)

    return train_x, train_y, paragraph_label


# rewrite one-hot binarizer
def binarizer(str_list, speakers_count_dict={}):
    count = 0
    for i in range(len(str_list)):
        if str_list[i] not in speakers_count_dict:
            print(str_list[i], 'not in dict', 'count', count)
            speakers_count_dict[str_list[i]] = count
            count += 1

        label_bin_arr = np.zeros((1, 25))
        label_bin_arr[0, speakers_count_dict[str_list[i]]] = 1

        if i == 0:
            result = label_bin_arr
        else:
            result = np.concatenate((result, label_bin_arr))

    return result


def delta(feat, N):
    NUMFRAMES = len(feat)
    # print("num frames: %d", NUMFRAMES)
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_feat = np.empty_like(feat)
    # pad the row
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
    for t in range(NUMFRAMES):
        # use partial finite difference to approximate the derivative
        delta_feat[t] = np.dot(np.arange(-N, N + 1), padded[t: t + 2 * N + 1]) / denominator
    return delta_feat


# this functions is used to count the number of audio clips of different users
def statistics(wav_files):
    statistics = {}
    for i, onewav in enumerate(wav_files):

        label = onewav.split("\\")[1].split('_')[0][1:]
        if label not in statistics:
            statistics[label] = 1
        else:
            statistics[label] = statistics[label] + 1

    return statistics


def res_unit(x, filters, pool=False, regularized=0.0):
    res = x
    if pool:
        x = MaxPool1D(2, padding="same")(x)
        res = Conv1D(filters=filters, kernel_size=1, strides=2, padding="same")(res)
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    if regularized > 0:
        out = Conv1D(filters=filters, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(regularized))(
            out)
    else:
        out = Conv1D(filters=filters, kernel_size=3, padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    if regularized > 0:
        out = Conv1D(filters=filters, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(regularized))(
            out)
    else:
        out = Conv1D(filters=filters, kernel_size=3, padding="same")(out)
    out = add([res, out])

    return out


def res_model(input_shape):
    mfcc_features = Input(input_shape)
    net = Conv1D(filters=32, kernel_size=4, padding="same")(mfcc_features)
    net = res_unit(net, 32, pool=True)
    net = res_unit(net, 32)
    net = res_unit(net, 32)

    net = res_unit(net, 64, pool=True)
    net = res_unit(net, 64)
    net = res_unit(net, 64)

    net = res_unit(net, 128, pool=True)
    net = res_unit(net, 128, regularized=0.1)
    net = res_unit(net, 128, regularized=0.1)

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.25)(net)

    net = AveragePooling1D(4)(net)
    net = Bidirectional(LSTM(256))(net)
    net = Dense(25, activation='softmax')(net)

    model = Model(inputs=mfcc_features, outputs=net)
    return model


def train(x_train, y_train, x_validate, y_validate):

    # class_dim = y_train.shape[1]
    # model = Sequential()
    # model.add(Conv1D(32, 4, input_shape=(1560, 39)))
    # model.add(MaxPool1D(4))
    # model.add(BatchNormalization())
    # model.add(Conv1D(64, 4))
    # model.add(MaxPool1D(4))
    # model.add(BatchNormalization())
    # model.add(Bidirectional(LSTM(256)))
    # model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dense(class_dim, activation='softmax'))

    model = res_model((1024, 39))

    # learning_rate_adjust = [
    #     CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)
    # ]
    print(model.summary())
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='D:\VoiceRecognition\ckt',
                                                    monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                                    mode='max', period=10)
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.0001), metrics=[categorical_accuracy])
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate),
                        callbacks=[checkpoint],
                        batch_size=32, epochs=100)

    return model, history


def matrix_build_pca(feature_data):
    matrix_list = []
    width = feature_data[0].shape[0] * feature_data[0].shape[1]
    for i in range(len(feature_data)):
        matrix_list.append(feature_data[i].flatten())

    matrix = np.array(matrix_list)
    np.savez('feature_matrix', matrix)
    return matrix


if __name__ == '__main__':
    # first train
    # wav_files = get_wav_files("D:/VoiceRecognition/data_thchs30/train_mixed")
    # x, y, paragraph_label = make_feature(wav_files)
    # np.savez('input_feature', x, y)

    # # after the first train
    speakers_count_dict = np.load('speakers_count_dict.npy', allow_pickle=True).item()
    input_feature = np.load('input_feature.npz')
    x = input_feature['arr_0']
    y = input_feature['arr_1']

    print(x.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

    # trained_model, history = train(x_train, y_train, x_validate, y_validate)
    # trained_model.save('D:/VoiceRecognition/cnnlstm')
    # epochs = range(len(history.history['categorical_accuracy']))
    # plt.figure()
    # plt.plot(epochs, history.history['categorical_accuracy'], 'b', label='categorical_accuracy')
    # plt.plot(epochs, history.history['val_categorical_accuracy'], 'r', label='val_categorical_accuracy')
    # plt.title("Accuracy on Training and Validation Data")
    # plt.xlabel("epochs")
    # plt.ylabel("accuracy")
    # plt.legend()
    # plt.savefig('acc.png', dpi=500, bbox_inches='tight')
    # plt.show()

    # plt.figure()
    # plt.plot(epochs, history.history['loss'], 'b', label='loss')
    # plt.plot(epochs, history.history['val_loss'], 'r', label='val_loss')
    # plt.title("Loss on Training and Validation Data")
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.legend()
    # plt.savefig('loss.png', dpi=500, bbox_inches='tight')
    # plt.show()

    # after the first time
    trained_model = load_model('D:/VoiceRecognition/ckt')
    print(trained_model.summary())
    loss, accuracy = trained_model.evaluate(x_test, y_test)
    print("Loss on test set:", loss, "Accuracy: ", accuracy)
