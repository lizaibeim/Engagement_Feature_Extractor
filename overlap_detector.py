import os
import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.optimizers import RMSprop, Adadelta, Adam
from cosine_annealing import CosineAnnealingScheduler
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)


class OverlapDetector:

    def __init__(self, images_dir, labels_path, *, imagetype='rgb'):
        """
        :param images_dir: absolute path of the directory storing the images
        :param labels_path: absolute path of the label xlsx file
        :param imagetype: the type of input images, gray or rgb
        """
        self.channel = 3 if imagetype == 'rgb' else 1
        self.images_dir = images_dir
        self.labels_path = labels_path
        self.__train_labels_path = None
        self.__train_images_dir = None
        self.__test_labels_path = None
        self.__test_images_dir = None
        self.__train_labels_path_aug = None
        self.__train_images_dir_aug = None
        self.__weights_factor = None
        self.__model_path = None
        self.__augmented = False
        self.__weighted = False
        self.__x = None
        self.__y = None
        self.__train_x = None
        self.__train_y = None
        self.__test_x = None
        self.__test_y = None
        self.split_train_test()

    @property
    def augmented(self):
        return self.__augmented

    @property
    def weighted(self):
        return self.__weighted

    @property
    def model_path(self):
        return self.__model_path

    @staticmethod
    def load_labels(label_path):
        """
        :param label_path: absolute path of the labels file (.xlsx)
        :return: np array of label
        """
        df = pd.read_excel(label_path)
        # sort labels by session, then segment in ascending order
        df.sort_values(by=['Sessions', 'Segments'], inplace=True)
        labels = np.array(df['Overlap'])

        return labels

    @staticmethod
    def load_images(image_dir, channels):
        """
        :param image_dir: absolute path of the mel-spectrogram images
        :param channels: image channels (gray-scale: 1, rgb: 3)
        :return: np array of image data
        """
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

    @staticmethod
    def weighted_categorical_crossentropy(weights):
        """
        :param weights: weights array
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

    @staticmethod
    def cal_weighted_penalty(labels):
        weight = np.zeros(3)  # assume 3 class
        for label in labels:
            weight[label] += 1

        base = np.amax(weight)
        for i in range(3):
            weight[i] = base / weight[i]

        return np.sqrt(weight)

    @staticmethod
    def augment_images(in_images_dir, in_labels_path, out_images_dir, out_labels_path):
        """
        :param in_images_dir:
        :param in_labels_path:
        :param out_images_dir:
        :param out_labels_path:
        :return:
        """
        if not os.path.isdir(out_images_dir):
            os.mkdir(out_images_dir)

        # read origin labels and sort
        df = pd.read_excel(in_labels_path)
        df.sort_values(by=['Sessions', 'Segments'], inplace=True)
        labels = np.array(df['Overlap'])

        # calculated the ratio for each class to duplicate
        count = [0, 0, 0]
        ratio = [0, 0, 0]
        for label in labels:
            count[label] += 1

        base = max(count)
        for m in range(3):
            ratio[m] = base / count[m] - 1

        # images path list for each class
        images_c0 = []
        images_c1 = []
        images_c2 = []
        images_classes_path = []

        names = os.listdir(in_images_dir)
        # sort images file by sessions, then segments in ascending order aligned with labels' order
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

        # duplicate original images to augment folder first
        for images_classes in images_classes_path:
            for name in images_classes:
                image_path = os.path.join(in_images_dir, name)
                src = cv.imread(image_path)
                os.chdir(out_images_dir)
                cv.imwrite(name, src)

        # traverse each class, augment images
        for k in range(3):
            images_classes = images_classes_path[k]

            # duplicate images for ratio[k] times
            for i in range(round(ratio[k])):

                for name in images_classes:
                    image_path = os.path.join(in_images_dir, name)
                    src = cv.imread(image_path)

                    # down-sampling image and then up-sampling for i+1 times
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
                    df = df.append([{'Sessions': session, 'Segments': int(new_segment), 'Overlap': k}],
                                   ignore_index=True)

        sorted_df = df.sort_values(by=['Sessions', 'Segments'], ascending=[True, True])
        sorted_df.to_excel(out_labels_path, index=None)

    def __augment_data(self, images_dir, labels_path):
        """
        :return: Augment images based on the ratio of each class
        """
        self.__train_labels_path_aug = labels_path[:-5] + 'Aug.xlsx'
        self.__train_images_dir_aug = images_dir + '_augmented'
        self.__augmented = True
        if os.path.exists(labels_path[:-5] + 'Aug.xlsx') and os.path.isdir(images_dir + '_augmented'):
            print("Already augment")
            return
        self.augment_images(images_dir, labels_path, self.__train_images_dir_aug, self.__train_labels_path_aug)
        print("Augment the images and labels done")

    def __load_model(self):
        assert self.__model_path is not None, 'You should train your initial model or populate an outer model into the overlap detector'
        if self.__weighted:
            pretrained_model = tf.keras.models.load_model(self.__model_path, custom_objects={'loss': self.weighted_categorical_crossentropy(self.__weights_factor)})
        else:
            pretrained_model = tf.keras.models.load_model(self.__model_path, custom_objects={'loss': 'categorical_crossentropy'})
        return pretrained_model

    def __train_model(self, x_train, y_train, x_validate, y_validate, input_shape, epochs, batch_size, weights_factor=None):
        class_dim = y_train.shape[1]
        model = tf.keras.Sequential()

        # first layer, the input shape equals to the shape of one image file, regardless of the batch size
        model.add(tf.keras.layers.Conv2D(16, 3, strides=1, activation="relu", input_shape=input_shape, name="conv16"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 2)))
        model.add(tf.keras.layers.Conv2D(8, (4, 1), activation="relu", name="freq_conv8"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(6, 2)))
        model.add(tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(class_dim)))
        model.add(tf.keras.layers.Dense(class_dim, activation='relu'))
        model.add(tf.keras.layers.Activation('softmax'))

        loss = self.weighted_categorical_crossentropy(weights_factor) if weights_factor is not None else "categorical_crossentropy"

        model.compile(loss=loss, optimizer=Adadelta(lr=0.001),
                      metrics=[tf.keras.metrics.categorical_accuracy, tf.keras.metrics.AUC()])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        learning_rate_adjust = [
            CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)
        ]

        print(model.summary())

        history = model.fit(x_train, y_train, validation_data=(x_validate, y_validate),
                            callbacks=[early_stopping, learning_rate_adjust],
                            batch_size=batch_size, epochs=epochs)
        return model, history

    def split_train_test(self):
        labels = self.load_labels(self.labels_path)
        print(labels.shape)
        if self.__y is None:
            self.__x = self.load_images(self.images_dir, self.channel)
            enc = OneHotEncoder()
            rlabels = np.reshape(labels, (-1, 1))
            print(rlabels.shape)
            self.__y = enc.fit_transform(rlabels).toarray()

        print(self.__x.shape)
        # kf = KFold(n_splits=5, shuffle=True, random_state=0)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        train_index, test_index = list(kf.split(self.__x, labels))[0]

        df = pd.read_excel(self.labels_path)
        # sort labels by session, then segment in ascending order
        df.sort_values(by=['Sessions', 'Segments'], inplace=True)

        train_df = df.iloc[train_index, :]
        test_df = df.iloc[test_index, :]
        self.__train_labels_path = self.labels_path[:-5] + 'Train.xlsx'
        self.__test_labels_path = self.labels_path[:-5] + 'Test.xlsx'
        train_df.to_excel(self.__train_labels_path, index=None)
        test_df.to_excel(self.__test_labels_path, index=None)

        names = os.listdir(self.images_dir)
        # sort images file by sessions, then segments in ascending order
        names.sort(key=lambda name: (int(name.split('.')[0].split('_')[0][1:3]), int(name.split('.')[0].split('_')[3])))
        # images_path = [os.path.join(self.images_dir, name) for name in names]

        self.__train_images_dir = self.images_dir + '_train'
        self.__test_images_dir = self.images_dir + '_test'
        if not os.path.isdir(self.__train_images_dir):
            os.mkdir(self.__train_images_dir)
        if not os.path.isdir(self.__test_images_dir):
            os.mkdir(self.__test_images_dir)

        for i in train_index:
            image_path = os.path.join(self.images_dir, names[i])
            # image_path = images_path[i]
            src = cv.imread(image_path)
            os.chdir(self.__train_images_dir)
            cv.imwrite(names[i], src)
        for j in test_index:
            # print(j)
            image_path = os.path.join(self.images_dir, names[j])
            src = cv.imread(image_path)
            os.chdir(self.__test_images_dir)
            cv.imwrite(names[j], src)

    def load_train_data(self, with_augmented=False):
        """
        :param with_augmented: whether load augmented data or not
        :return: assign x, y data set to overlap detector
        """
        load_images_dir = self.__train_images_dir
        load_labels_path = self.__train_labels_path
        if with_augmented:
            # check whether the augmented images and labels keep aligned with the original images and labels
            # assert self.__augmented, 'You should augment the data first by calling function augment_data()'
            self.__augment_data(load_images_dir, load_labels_path)
            load_images_dir = self.__train_images_dir_aug
            load_labels_path = self.__train_labels_path_aug

        self.__train_x = self.load_images(load_images_dir, self.channel)
        labels = self.load_labels(load_labels_path)
        self.__weights_factor = self.cal_weighted_penalty(labels)
        enc = OneHotEncoder()
        labels = np.reshape(labels, (-1, 1))
        self.__train_y = enc.fit_transform(labels).toarray()

    def load_test_data(self):
        self.__test_x = self.load_images(self.__test_images_dir, self.channel)
        labels = self.load_labels(self.__test_labels_path)
        enc = OneHotEncoder()
        labels = np.reshape(labels, (-1, 1))
        self.__test_y = enc.fit_transform(labels).toarray()

    def train_model(self, save_path, epochs, batch_size, weighted=False, augmented=False):
        """
        :return: train your own overlap detector for the first time and save it
        """
        self.load_train_data(augmented)
        self.load_test_data()
        # x_train, x_val, y_train, y_val = self.__train_x, self.__test_x, self.__train_y, self.__test_y  # assume test set as validation set
        x_train, x_val, y_train, y_val = train_test_split(self.__train_x, self.__train_y, test_size=0.2, random_state=0, stratify=self.__train_y)

        input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        print('input shape', x_train.shape)
        self.__weighted = weighted
        weights_factor = self.__weights_factor if weighted else None
        print('weight ', weights_factor)
        trained_model, history = self.__train_model(x_train, y_train, x_val, y_val, input_shape, epochs,
                                                    batch_size, weights_factor)
        trained_model.save(save_path, save_format='tf')
        self.__model_path = save_path
        epochs = range(len(history.history['categorical_accuracy']))
        plt.figure()
        plt.plot(epochs, history.history['categorical_accuracy'], 'b', label='categorical_accuracy')
        plt.plot(epochs, history.history['val_categorical_accuracy'], 'r', label='val_categorical_accuracy')
        plt.legend()
        plt.savefig("D:\SpectrumAnalysis\\"+self.__model_path.split('\\')[-1])
        # return trained_model

    def populate_model(self, model_path, weighted, augmented):
        """
        :param model_path: populate a previous model created by this overlap detector will overwrite the current referred model
        :param weight: you need to specific whether the model you want to populate is weighted penalized or not
        :return:
        """
        self.__weighted = weighted
        self.__model_path = model_path
        self.load_train_data(augmented)
        self.load_test_data()

    def continue_train_model(self, save_path, epochs, batch_size, weighted=False, augmented=False):
        pretrained_model = self.__load_model()

        # early_stopping = EarlyStopping(monitor='val_categorical_accuracy', mode='max', patience=10)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        learning_rate_adjust = [
            CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)
        ]
        # x_train, x_val, y_train, y_val = train_test_split(self.__x, self.__y, test_size=0.3, random_state=0,
        #                                                   stratify=self.__y)
        x_train, x_val, y_train, y_val = train_test_split(self.__train_x, self.__train_y, test_size=0.2, random_state=0,
                                                          stratify=self.__train_y)
        pretrained_model.fit(x_train, y_train, validation_data=(x_val, y_val),
                             callbacks=[early_stopping, learning_rate_adjust],
                             batch_size=batch_size, epochs=epochs)
        if save_path is None:
            pretrained_model.save(save_path, save_format='tf')
        else:
            pretrained_model.save(self.__model_path, save_format='tf')

        return pretrained_model

    def evaluation_on_val(self):
        pretrained_model = self.__load_model()
        x_val, y_val = self.__test_x, self.__test_y  # assume test set as validation set
        # x_train, x_val, y_train, y_val = train_test_split(self.__train_x, self.__train_y, test_size=0.2, random_state=0,
        #                                                   stratify=self.__train_y)

        predictions_class = pretrained_model.predict_classes(x_val)
        predictions_prob = pretrained_model.predict(x_val)

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

        # for i in range(len(predictions_prob)):
        #     print(predictions_prob[i], y_val[i])

    def predict(self):
        pretrained_model = self.__load_model()
        predictions_class = pretrained_model.predict_classes(self.__x)
        predictions_prob = pretrained_model.predict(self.__x)

        confusion_matrix = np.zeros((3, 3))
        for i in range(predictions_prob.shape[0]):
            index_row = np.where(self.__y[i] == np.amax(self.__y[i]))
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

        for i in range(len(predictions_prob)):
            print("segment", i, " ", predictions_prob[i], self.__y[i])
