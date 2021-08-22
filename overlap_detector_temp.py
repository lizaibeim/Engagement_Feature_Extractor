import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.python.keras.models import Model

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)
import os
import itertools
from tensorflow.python.framework import ops
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import RMSprop, Adadelta, Adam
from cosine_annealing import CosineAnnealingScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, add, Input, LSTM, Dense, Activation, Conv2D, MaxPool2D, Flatten, \
    Dropout, BatchNormalization, Lambda, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def label_loader(label_path):
    '''
    :param path: absolute path of the labels file (.xlsx)
    :return: np array of label
    '''
    df = pd.read_excel(label_path)
    # sort labels by session, then segment in ascending order
    df.sort_values(by=['Sessions', 'Segments'], inplace=True)
    labels = np.array(df['Overlap'])

    return df, labels


def image_loader(image_dir, channels):
    '''
    :param image_dir: absolute path of the mel-spectrogram images
    :return: np array of image data
    '''
    names = os.listdir(image_dir)
    # sort images file by sessions, then segments in ascending order
    names.sort(key=lambda name: (int(name.split('.')[0].split('_')[0][1:3]), int(name.split('.')[0].split('_')[3])))
    images_path = [os.path.join(image_dir, name) for name in names]

    images_data = []
    for image_file in images_path:
        # print(image_file)
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image, channels)
        images_data.append(image)

    # convert to tensor with shape (samples, image.shape[0], image.shape[1])
    features_data = tf.stack(images_data, axis=0)

    return features_data.numpy()


# augment the image data by downsampling and upsampling
def data_augmentation(images_dir, labels_path, out_images_dir, out_labels_path):
    '''
    :param images: augment images to balanced the dataset
    :return:
    '''

    if not os.path.isdir(out_images_dir):
        os.mkdir(out_images_dir)

    df = pd.read_excel(labels_path)
    df.sort_values(by=['Sessions', 'Segments'], inplace=True)
    labels = np.array(df['Overlap'])

    # calculated the ratio
    count = [0, 0, 0]
    ratio = [0, 0, 0]
    for label in labels:
        count[label] += 1
    print("Count: ", count)
    base = max(count)
    ratio[0] = base / count[0] - 1
    ratio[1] = base / count[1] - 1
    ratio[2] = base / count[2] - 1
    print("scale ratio: ", ratio)

    # images path list for each class
    images_c0 = []
    images_c1 = []
    images_c2 = []
    images_classes_path = []

    names = os.listdir(images_dir)
    # sort images file by sessions, then segments in ascending order to keep line with labels
    names.sort(key=lambda x: (int(x.split('.')[0].split('_')[0].replace('S', '')), int(x.split('.')[0].split('_')[3])))

    for i in range(len(df)):

        if df.iloc[i]['Overlap'] == 0:
            images_c0.append(names[i])
        elif df.iloc[i]['Overlap'] == 1:
            images_c1.append(names[i])
        elif df.iloc[i]['Overlap'] == 2:
            images_c2.append(names[i])
    images_classes_path.append(images_c0)
    images_classes_path.append(images_c1)
    images_classes_path.append(images_c2)

    # duplicate images to augment folder
    for images_classes in images_classes_path:
        for name in images_classes:
            image_path = os.path.join(images_dir, name)
            src = cv.imread(image_path)
            # crop = src[:, :-1]  # crop the size to (128, 150)
            os.chdir(out_images_dir)
            cv.imwrite(name, src)

    # augment image
    for k in range(3):
        images_classes = images_classes_path[k]
        print(images_classes_path[k])
        print("k is", k, " times to duplicate ", round(ratio[k]))
        for i in range(round(ratio[k])):
            for name in images_classes:

                image_path = os.path.join(images_dir, name)
                src = cv.imread(image_path)

                for j in range(i + 1):
                    src = cv.pyrDown(src)
                    src = cv.pyrUp(src)

                image_augmented = src[:, :-1]
                session = name.split('.')[0].split('_')[0]
                segment = name.split('.')[0].split('_')[3]

                new_segment = str(1000 + int(segment)) + str(i)
                write_name = session + '_audio_MONO_' + new_segment + '_16000_split.png'

                os.chdir(out_images_dir)
                cv.imwrite(write_name, image_augmented)

                df = df.append([{'Sessions': session, 'Segments': int(new_segment), 'Overlap': k}], ignore_index=True)

    sorted_df = df.sort_values(by=['Sessions', 'Segments'], ascending=[True, True])
    sorted_df.to_excel(out_labels_path, index=None)

    return 1


def weighted_categorical_crossentropy(weights):
    """
    :param weights:
    :return:
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)

        return loss

    return loss


def cal_weighted_penalty(labels, n_classes):
    quantity = np.zeros(n_classes)
    for label in labels:
        index = label - 1 if n_classes == 2 else label
        quantity[index] += 1

    print("number of each class: ", quantity)
    weights = np.zeros(n_classes)
    for i in range(n_classes):
        weights[i] = 1 - (quantity[i] / np.sum(quantity))

    return weights


def plot_confusion_matrix(cm,
                          target_names,
                          name,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(name, dpi=500, bbox_inches='tight')
    # plt.show()


def res_block(x, filters, pool=False, regularized=0.0):
    res = x
    if pool:
        res = Conv2D(filters=filters, kernel_size=1, strides=2, padding="same")(res)

    out = BatchNormalization()(x)
    out = Activation("elu")(out)
    if regularized > 0:
        out = Conv2D(filters=filters, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(regularized))(
            out)
    else:
        out = Conv2D(filters=filters, kernel_size=3, padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("elu")(out)
    if regularized > 0:
        out = Conv2D(filters=filters, kernel_size=(4, 1), padding="same",
                     kernel_regularizer=regularizers.l2(regularized))(out)
    else:
        out = Conv2D(filters=filters, kernel_size=(4, 1), padding="same")(out)
    if pool:
        out = MaxPool2D(pool_size=2, padding="same")(out)
    out = add([res, out])

    return out


def ResLSTM(input_shape, class_dim):
    input_features = Input(input_shape)
    net = Conv2D(filters=16, kernel_size=1, padding="same")(input_features)

    net = res_block(net, 32, pool=True)
    net = res_block(net, 32)
    net = res_block(net, 32)

    net = res_block(net, 64, pool=True)
    net = res_block(net, 64)
    net = res_block(net, 64)

    net = res_block(net, 128, pool=True)
    net = res_block(net, 128)
    net = res_block(net, 128)

    net = Lambda(lambda x: K.mean(x, axis=1))(net)
    net = Bidirectional(LSTM(256))(net)
    net = Dropout(0.25)(net)
    net = LeakyReLU()(net)
    net = Dense(class_dim, activation="softmax")(net)
    model = Model(inputs=input_features, outputs=net)

    return model


def train_model(x_train, y_train, x_validate, y_validate, input_shape, epochs, batch_size, weights=None):
    class_dim = y_train.shape[1]
    # model = tf.keras.Sequential()
    # initializer = tf.keras.initializers.HeNormal(seed=seed)

    # first layer, the input shape equals to the shape of one image file, regardless of the batch size
    # model.add(tf.keras.layers.Conv2D(16, 3, strides=1, activation="elu", input_shape=input_shape, name="conv16"))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 2)))
    # model.add(tf.keras.layers.Conv2D(8, (4, 1), activation="elu", name="freq_conv8"))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(6, 2)))
    # model.add(tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1)))
    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)))
    # model.add(tf.keras.layers.Dense(256))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.LeakyReLU())
    # model.add(tf.keras.layers.Dense(class_dim, activation='softmax'))

    model = ResLSTM(input_shape, class_dim)

    if class_dim == 2:
        loss = "binary_crossentropy"

    if class_dim > 2:
        loss = "categorical_crossentropy"

    if weights is not None:
        print("weighted penalized")
        loss = weighted_categorical_crossentropy(weights)

    model.compile(loss=loss, optimizer=Adadelta(lr=0.001),
                  metrics=[tf.keras.metrics.categorical_accuracy, tf.keras.metrics.AUC()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    learning_rate_adjust = [
        CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)
    ]
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='D:\SpectrumAnalysis\ckt',
                                                    monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                                    mode='max', period=1)

    print(model.summary())
    history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate),
                        callbacks=[early_stopping, learning_rate_adjust, checkpoint],
                        batch_size=batch_size, epochs=epochs)
    return model, history


if __name__ == '__main__':
    labels_path = "D:\SpectrumAnalysis\labelling_results\OverlapLabels.xlsx"
    images_dir = "D:\SpectrumAnalysis\mel_spectrum_zcr"
    # images_dir = "D:\SpectrumAnalysis\mel_spectrum_viridis"
    # images_dir = "D:\SpectrumAnalysis\mel_spectrum_gray"

    # augment the data
    aug_labels_path = "D:\SpectrumAnalysis\labelling_results\OverlapLabelsAug.xlsx"
    aug_images_dir = "D:\SpectrumAnalysis\mel_spectrum_zcr_augmented"
    # aug_images_dir = "D:\SpectrumAnalysis\mel_spectrum_viridis_augmented"
    # aug_images_dir = "D:\SpectrumAnalysis\mel_spectrum_gray_augmented"
    # data_augmentation(images_dir, labels_path, aug_images_dir, aug_labels_path)

    # test data
    test_labels_path = "D:\SpectrumAnalysis\labelling_results\OverlapLabels_outer_test.xlsx"
    test_images_dir = "D:\SpectrumAnalysis\mel_spectrum_zcr_test"
    # test_images_dir = "D:\SpectrumAnalysis\mel_spectrum_viridis_test"
    # test_images_dir = "D:\SpectrumAnalysis\mel_spectrum_gray_test"

    # read labels and images in np.array
    # _, labels = label_loader(aug_labels_path)
    _, labels = label_loader(labels_path)
    # x = image_loader(aug_images_dir, 3)
    # x = image_loader(aug_images_dir, 1)
    # x = image_loader(images_dir, 1)
    x = image_loader(images_dir, 3)

    _, test_labels = label_loader(test_labels_path)
    test_x = image_loader(test_images_dir, 3)
    # test_x = image_loader(test_images_dir, 1)
    weights = None

    # create weight vector for imbalanced dataset
    weights = cal_weighted_penalty(labels, 3)
    print("weights ", weights)
    print('x shape: ', x.shape)
    print('labels shape: ', labels.shape)

    enc = OneHotEncoder()
    labels = np.reshape(labels, (-1, 1))
    y = enc.fit_transform(labels).toarray()

    test_labels = np.reshape(test_labels, (-1, 1))
    test_y = enc.fit_transform(test_labels).toarray()

    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
    x_train, x_val, y_train, y_val = x, test_x, y, test_y

    # ############################################# START TRAIN MODEL ###################################################
    trained_model, history = train_model(x_train, y_train, x_val, y_val, (128, 151, 3), epochs=60, batch_size=16,
                                         weights=weights)

    # epochs = range(len(history.history['categorical_accuracy']))
    # plt.figure()
    # plt.plot(epochs, history.history['categorical_accuracy'], 'b', label='categorical_accuracy')
    # plt.plot(epochs, history.history['val_categorical_accuracy'], 'r', label='val_categorical_accuracy')
    # plt.legend()
    # plt.show()

    # with open('D:/SpectrumAnalysis/training_history_zcr_augmented_weighted_60', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)

    ########################################### LOAD PRE TRAINED MODEL #################################################
    # load_model = 'D:/SpectrumAnalysis/ckt'
    # trained_model = tf.keras.models.load_model(load_model,
    #                                            custom_objects={'loss': weighted_categorical_crossentropy(weights)})
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # learning_rate_adjust = [
    #     CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)
    # ]
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='D:\SpectrumAnalysis\ckt',
    #                                                 monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
    #                                                 mode='max', period=30)
    # trained_model.fit(x_train, y_train, validation_data=(x_val, y_val),
    #                   callbacks=[early_stopping, learning_rate_adjust, checkpoint],
    #                   batch_size=16, epochs=100)
    # trained_model.save('D:\SpectrumAnalysis\models\cnnlstm_zcr_3classes_augmented_weighted_600epochs', save_format='tf')
    # predictions_class = trained_model.predict_classes(x_val)
    # predictions_prob = trained_model.predict(x_val)

    ####################################################CHECK POINT TEST######################################################
    load_model = 'D:/SpectrumAnalysis/ckt'
    # load_model = 'D:/SpectrumAnalysis/models/reslstm_zcr_3classes_unaugmented_weighted_60epochs'

    trained_model = tf.keras.models.load_model(load_model, custom_objects={'loss': weighted_categorical_crossentropy(weights)})

    # trained_model = tf.keras.models.load_model(load_model)
    # trained_model.save('D:/SpectrumAnalysis/models/cldnn_zcr_3classes_augmented_weighted_60epochs',
    #                    save_format='tf')

    # predictions_class = trained_model.predict_classes(x_val)
    predictions_class = np.argmax(trained_model.predict(x_val), axis=1)
    predictions_prob = trained_model.predict(x_val)

    confusion_matrix = np.zeros((3, 3))
    for i in range(predictions_prob.shape[0]):
        index_row = np.where(y_val[i] == np.amax(y_val[i]))
        index_col = int(predictions_class[i])
        confusion_matrix[index_row, index_col] += 1

    tp = confusion_matrix[2, 2]
    fn = confusion_matrix[2, 0] + confusion_matrix[2, 1]
    fp = confusion_matrix[0, 2] + confusion_matrix[1, 2]

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    # rows = ground truth; cols = predicted
    print(confusion_matrix)
    print('Overlapped recall: ', recall, " precision: ", precision)

    predictions_prob_test = trained_model.predict(test_x)
    # predictions_class_test = trained_model.predict_classes(test_x)
    predictions_class_test = np.argmax(trained_model.predict(test_x), axis=1)
    for i in range(len(predictions_prob_test)):
        print(i, predictions_prob_test[i], test_y[i])

    confusion_matrix = np.zeros((3, 3))
    for i in range(predictions_prob_test.shape[0]):
        index_row = np.where(test_y[i] == np.amax(test_y[i]))
        index_col = int(predictions_class_test[i])
        confusion_matrix[index_row, index_col] += 1

    tp = confusion_matrix[2, 2]
    fn = confusion_matrix[2, 0] + confusion_matrix[2, 1]
    fp = confusion_matrix[0, 2] + confusion_matrix[1, 2]

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    # rows = ground truth; cols = predicted
    print(confusion_matrix)
    print('Overlapped recall: ', recall, " precision: ", precision)

    # plot_confusion_matrix(cm=np.array(confusion_matrix),
    #                       normalize=False,
    #                       name='Confusion_Matrix_ZCR_Augmented_Weighted.png',
    #                       target_names=['silent', 'non-overlapped', 'overlapped'],
    #                       title="Confusion Matrix Augmented Weighted(ZCR)"
    #                       )

    # with open('training_history_gray_unaugmented_unweighted_60', 'rb') as file_pi:
    #     history1 = pickle.load(file_pi)
    # with open('training_history_viridis_unaugmented_unweighted_60', 'rb') as file_pi:
    #     history2 = pickle.load(file_pi)
    # with open('training_history_zcr_unaugmented_unweighted_60', 'rb') as file_pi:
    #     history3 = pickle.load(file_pi)
    # with open('training_history_gray_unaugmented_weighted_60', 'rb') as file_pi:
    #     history4 = pickle.load(file_pi)
    # with open('training_history_viridis_unaugmented_weighted_60', 'rb') as file_pi:
    #     history5 = pickle.load(file_pi)
    # with open('training_history_zcr_unaugmented_weighted_60', 'rb') as file_pi:
    #     history6 = pickle.load(file_pi)
    # with open('training_history_gray_augmented_weighted_60', 'rb') as file_pi:
    #     history7 = pickle.load(file_pi)
    # with open('training_history_viridis_augmented_weighted_60', 'rb') as file_pi:
    #     history8 = pickle.load(file_pi)
    # with open('training_history_zcr_augmented_weighted_60', 'rb') as file_pi:
    #     history9 = pickle.load(file_pi)
    #
    # # plt.plot(history1['loss'])
    # # print(history1['val_loss'])
    # plt.plot(np.arange(len(history1['val_loss'])), history1['val_loss'], '-.', color='c')
    # plt.plot(np.arange(len(history2['val_loss'])), history2['val_loss'], '-.', color='b')
    # plt.plot(np.arange(len(history3['val_loss'])), history3['val_loss'], '-.', color='g')
    # plt.plot(np.arange(len(history4['val_loss'])), history4['val_loss'], '-.', color='r')
    # plt.plot(np.arange(len(history5['val_loss'])), history5['val_loss'], '-.', color='m')
    # plt.plot(np.arange(len(history6['val_loss'])), history6['val_loss'], '-.', color='y')
    # plt.plot(np.arange(len(history7['val_loss'])), history7['val_loss'], '-.', color='k')
    # plt.plot(np.arange(len(history8['val_loss'])), history8['val_loss'], '-.', color=(0, 1, 0))
    # plt.plot(np.arange(len(history9['val_loss'])), history9['val_loss'], '-.', color=(0.92, 0.98, 0))
    #
    # plt.title('Model Validation Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['UAG-UWT-G', 'UAG-UWT-V', 'UAG-UWT-Z', 'UAG-WT-G', 'UAG-WT-V', 'UAG-WT-Z', 'AG-WT-G', 'AG-WT-V', 'AG-WT-Z'], loc='upper right')
    # plt.savefig('val_loss.png', dpi=500, bbox_inches='tight')
    # plt.show()
    #
    # with open('training_history_gray_unaugmented_unweighted_60', 'rb') as file_pi:
    #     history1 = pickle.load(file_pi)
    # with open('training_history_viridis_unaugmented_unweighted_60', 'rb') as file_pi:
    #     history2 = pickle.load(file_pi)
    # with open('training_history_zcr_unaugmented_unweighted_60', 'rb') as file_pi:
    #     history3 = pickle.load(file_pi)
    # with open('training_history_gray_unaugmented_weighted_60', 'rb') as file_pi:
    #     history4 = pickle.load(file_pi)
    # with open('training_history_viridis_unaugmented_weighted_60', 'rb') as file_pi:
    #     history5 = pickle.load(file_pi)
    # with open('training_history_zcr_unaugmented_weighted_60', 'rb') as file_pi:
    #     history6 = pickle.load(file_pi)
    # with open('training_history_gray_augmented_weighted_60', 'rb') as file_pi:
    #     history7 = pickle.load(file_pi)
    # with open('training_history_viridis_augmented_weighted_60', 'rb') as file_pi:
    #     history8 = pickle.load(file_pi)
    # with open('training_history_zcr_augmented_weighted_60', 'rb') as file_pi:
    #     history9 = pickle.load(file_pi)
    #
    # # plt.plot(history1['loss'])
    # # print(history1['val_loss'])
    # plt.plot(np.arange(len(history1['loss'])), history1['loss'], '-.', color='c')
    # plt.plot(np.arange(len(history2['loss'])), history2['loss'], '-.', color='b')
    # plt.plot(np.arange(len(history3['loss'])), history3['loss'], '-.', color='g')
    # plt.plot(np.arange(len(history4['loss'])), history4['loss'], '-.', color='r')
    # plt.plot(np.arange(len(history5['loss'])), history5['loss'], '-.', color='m')
    # plt.plot(np.arange(len(history6['loss'])), history6['loss'], '-.', color='y')
    # plt.plot(np.arange(len(history7['loss'])), history7['loss'], '-.', color='k')
    # plt.plot(np.arange(len(history8['loss'])), history8['loss'], '-.', color=(0, 1, 0))
    # plt.plot(np.arange(len(history9['loss'])), history9['loss'], '-.', color=(0.92, 0.98, 0))
    #
    # plt.title('Model Train Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['UAG-UWT-G', 'UAG-UWT-V', 'UAG-UWT-Z', 'UAG-WT-G', 'UAG-WT-V', 'UAG-WT-Z', 'AG-WT-G', 'AG-WT-V', 'AG-WT-Z'], loc='upper right')
    # plt.savefig('train_loss.png', dpi=500, bbox_inches='tight')
    # plt.show()
