import pandas as pd
import wave
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import find_peaks
from pylab import *


def cal_volume(wave_data, window_size, over_lap):
    wlen = len(wave_data)
    step = window_size - over_lap
    frame_num = int(math.ceil(wlen * 1.0 / step))
    volume = np.zeros((frame_num, 1))

    for i in range(frame_num):
        cur_frame = wave_data[np.arange(i * step, min(i * step + window_size, wlen))]
        cur_frame = cur_frame - np.median(cur_frame)
        volume[i] = np.sum(np.abs(cur_frame))

    return volume


def cal_volume_db(wave_data, window_size, over_lap):
    wlen = len(wave_data)
    step = window_size - over_lap
    frame_num = int(math.ceil(wlen * 1.0 / step))
    decibel = np.zeros((frame_num, 1))

    for i in range(frame_num):
        cur_frame = wave_data[np.arange(i * step, min(i * step + window_size, wlen))]
        cur_frame = cur_frame - np.mean(cur_frame)
        decibel[i] = 10 * np.log10(np.sum(cur_frame * cur_frame))

    return decibel


def cal_pause_time(volume_data):
    wlen = len(volume_data)
    pause = 0

    for i in range(wlen):
        if abs(volume_data[i]) <= 20000:
            pause += 1
    pause_percent = pause / wlen * 100

    return pause_percent


def cal_silence_value():
    path = './segments_mono_10/S02_audio_MONO/S02_audio_MONO_20_16000_split.wav'
    f = wave.open(path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    start = round(framerate * 7)
    end = round(framerate * 8)
    silence_part_vol = cal_volume(wave_data[start:end], 256, 128)
    # print(silence_part_vol)
    return np.average(silence_part_vol)


def draw_variable_importance(feature_list, importances):
    plt.style.use('fivethirtyeight')  # style
    x_values = list(range(len(importances)))  # list of x locations
    plt.bar(x_values, importances, orientation='vertical')  # make bar chart
    plt.yticks(fontsize=10)
    plt.xticks(x_values, feature_list, rotation='vertical', fontsize=10)  # tick labels for x axis
    plt.ylabel('Importance', fontsize=10)
    plt.xlabel('Feature', fontsize=10)
    plt.title('Feature Importances', fontsize=20)

    for x, y in zip(x_values, importances):
        plt.text(x, y + 0.001, '%.2f' % y, ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig('Variable_importance.png', dpi=500, bbox_inches='tight')


def draw_wave_volume_decibel(wave_data, volume_data, decibel_data, path, nframes, framerate, window_size, over_lap,
                             volume_peaks, decibel_peaks):
    fig = plt.figure()
    time = np.arange(0, nframes) * (1.0 / framerate)
    time2 = np.arange(0, len(volume_data)) * (window_size - over_lap) * 1.0 / framerate
    volume_peaks_indices = [time2[i] for i in volume_peaks]
    decibel_peaks_indices = [time2[i] for i in decibel_peaks]

    plt.subplot(311)
    plt.plot(time, wave_data)
    plt.ylabel("Amplitude")
    plt.subplot(312)
    plt.plot(time2, volume_data)
    plt.plot(volume_peaks_indices, volume_data[volume_peaks], '.', markersize=1, color='r')
    plt.ylabel("Volume")
    plt.subplot(313)
    plt.plot(time2, decibel_data, c="g")
    plt.plot(decibel_peaks_indices, decibel_data[decibel_peaks], '.', markersize=1, color='r')
    plt.ylabel("Decibel(dB)")
    plt.xlabel("time (seconds)")
    plt.tight_layout()
    plt.savefig('./volume_decibel/' + path[:-4] + '.png', dpi=500, bbox_inches='tight')
    plt.close('all')


def ternary_convert(predictions):
    for i in range(len(predictions)):
        if predictions[i] < 1.7:
            predictions[i] = 1
        elif predictions[i] >= 1.7 and predictions[i] <= 2.3:
            predictions[i] = 2
        else:
            predictions[i] = 3
    return predictions


def emotion_feature_gen(window_size=256, over_lap=128, norm=False):
    """
  :param window_size: Each window containing 256 sampling frames to calculate the volume or decibel
  :param over_lap: The next window will be overlapping with the last, over_lap = window_size - step_size
  :param norm: whether normalize the feature or not
  :return: generate a new xlsx file containing the emotion features as well as the labelling result
  """

    # should be in order with the labelling result
    src_dir = ['./segments_mono_10/S02_audio_MONO', './segments_mono_10/S03_audio_MONO',
               './segments_mono_10/S09_audio_MONO', './segments_mono_10/S20_audio_MONO',
               './segments_mono_10/S13_audio_MONO']

    volume_lst = [];
    decibel_lst = [];
    var_volume_lst = []
    std_volume_lst = [];
    var_decibel_lst = [];
    std_decibel_lst = []
    session_lst = [];
    segment_lst = [];
    pause_lst = []

    # traverse session
    for i in range(len(src_dir)):
        sub_dir = src_dir[i]
        files_name = os.listdir(sub_dir)
        files_path = [sub_dir + "\\" + f for f in files_name if f.endswith('.wav')]

        for i in range(len(files_path)):
            file_path = files_path[i]
            session = files_name[i].split(sep='_')[0]
            segment = int(files_name[i].split(sep='_')[3])

            f = wave.open(file_path, 'rb')
            params = f.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]

            str_data = f.readframes(nframes)
            wave_data = np.frombuffer(str_data, dtype=np.short)

            wave_data = wave_data * 1.0 / max(abs(wave_data)) if norm else wave_data  # normalization

            volume_arr = cal_volume(wave_data, window_size, over_lap)
            decibel_arr = cal_volume_db(wave_data, window_size, over_lap)
            pause_percent = cal_pause_time(volume_arr)
            volume = np.average(volume_arr)
            decibel = np.average(decibel_arr)

            # flatten() to convert the volume array to 1D array from shape (n,1) to (1, n)
            volume_peaks, _ = find_peaks(volume_arr.flatten(), height=None, threshold=None, distance=10,
                                         prominence=20000, width=None, wlen=None, rel_height=None,
                                         plateau_size=None)

            decibel_peaks, _ = find_peaks(decibel_arr.flatten(), height=None, threshold=None, distance=10,
                                          prominence=5, width=None, wlen=None, rel_height=None,
                                          plateau_size=None)

            volume_peaks_val = volume_arr[volume_peaks]
            decibel_peaks_val = decibel_arr[decibel_peaks]

            # draw_wave_volume_decibel(wave_data, volume_arr, decibel_arr, files_name[i], nframes, framerate, window_size, over_lap, volume_peaks, decibel_peaks)
            session_lst.append(session)
            segment_lst.append(segment)
            volume_lst.append(volume)
            decibel_lst.append(decibel)
            pause_lst.append(pause_percent)
            var_volume_lst.append(np.var(volume_peaks_val))
            std_volume_lst.append(np.std(volume_peaks_val, ddof=1))
            var_decibel_lst.append(np.var(decibel_peaks_val))
            std_decibel_lst.append(np.std(decibel_peaks_val, ddof=1))

    label_folder = r'D:\SpectrumAnalysis\labelling_results'
    files_name = os.listdir(label_folder)
    segments_coding_xlsv = [f for f in files_name if (f.endswith('.xlsx') and f.startswith('segments_coding'))]
    segments_coding_path = os.path.join(label_folder, segments_coding_xlsv[0])

    df = pd.read_excel(segments_coding_path)

    data = {
        'Sessions': session_lst,
        'Segments': segment_lst,
        'Volume': volume_lst,
        'Decibel': decibel_lst,
        'Volume_var': var_volume_lst,
        'Decibel_var': var_decibel_lst,
        'Volume_std': std_volume_lst,
        'Decibel_std': std_decibel_lst,
        'Pause_percent': pause_lst
    }

    new_df = pd.DataFrame(data, columns=['Sessions', 'Segments', 'Volume', 'Decibel', 'Volume_var', 'Decibel_var',
                                         'Volume_std', 'Decibel_std', 'Pause_percent'])
    new_df = pd.merge(new_df, df, how='outer')
    sorted_df = new_df.sort_values(by=['Sessions', 'Segments'], ascending=[True, True])

    out_excel_path = os.path.join(label_folder, 'emotion_dataset_norm.xlsx') if norm else os.path.join(label_folder,
                                                                                                       'emotion_dataset.xlsx')
    sorted_df.to_excel(out_excel_path, index=None)


def regressor_evaluation(df_features):
    # Volume, Decibel, Volume_var, Decibel_var, Volume_std, Decibel_std, Pause_percent
    feature_checkbox = [0, 0, 0, 0, 0, 0, 0]
    for col in df_features.columns:
        if col == 'Volume':
            feature_checkbox[0] = 1
        elif col == 'Decibel':
            feature_checkbox[1] = 1
        elif col == 'Volume_var':
            feature_checkbox[2] = 1
        elif col == 'Decibel_var':
            feature_checkbox[3] = 1
        elif col == 'Volume_std':
            feature_checkbox[4] = 1
        elif col == 'Decibel_std':
            feature_checkbox[5] = 1
        elif col == 'Pause_percent':
            feature_checkbox[6] = 1

    # retrieve emotion labels
    emotion_labels = np.array(df_features['Emotion'])

    # remove labels from features
    df_features = df_features.drop('Emotion', axis=1)

    # convert to numpy array
    features = np.array(df_features)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    labels_lst = []
    pair_lst = []
    bias_lst = []
    mae_lst = []
    mape_lst = []
    rmse_lst = []

    for train_index, test_index in kf.split(features):
        # print('train index: ', train_index)
        # print('test index: ', test_index)
        train_features, train_labels = features[train_index], emotion_labels[train_index]
        test_features, test_labels = features[test_index], emotion_labels[test_index]

        rf = RandomForestRegressor(n_estimators=1000, random_state=0)
        rf.fit(train_features, train_labels)
        predictions = rf.predict(test_features)

        labels_lst = labels_lst + list(test_labels)
        pair_lst = pair_lst + list(zip(predictions, test_labels))

    for pair in pair_lst:
        bias = pair[0] - pair[1]
        mae = abs(pair[0] - pair[1])
        mape = (abs(pair[0] - pair[1]) / pair[1])
        rmse = np.square(bias)

        bias_lst.append(bias)
        mape_lst.append(mape)
        mae_lst.append(mae)
        rmse_lst.append(rmse)

    overall_bias = np.mean(bias_lst)
    overall_mape = np.mean(mape_lst)
    overall_mae = np.mean(mae_lst)
    overall_rmse = np.mean(rmse_lst)

    # add bias, mape, mae, rmse into the feature_checkbox
    feature_checkbox.append(overall_bias)
    feature_checkbox.append(overall_mape)
    feature_checkbox.append(overall_mae)
    feature_checkbox.append(overall_rmse)
    feature_checkbox.append(abs(overall_bias) + overall_mape + overall_mae + overall_rmse)

    return feature_checkbox


def draw_prediction_results(labels_lst, predictions_lst, index_lst):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    seg_index1 = []
    seg_index2 = []
    seg_index3 = []
    for i in range(len(labels_lst)):
        if labels_lst[i] == 1:
            y1.append(predictions_lst[i])
            x1.append(labels_lst[i])
            seg_index1.append(index_lst[i])
        elif labels_lst[i] == 2:
            y2.append(predictions_lst[i])
            x2.append(labels_lst[i])
            seg_index2.append(index_lst[i])
        else:
            y3.append(predictions_lst[i])
            x3.append(labels_lst[i])
            seg_index3.append(index_lst[i])

    plt.figure(dpi=300)
    plt.plot(seg_index1, y1, 'r.', label='low emotion', markersize=1)
    plt.plot(seg_index2, y2, 'g.', label='neutral emotion', markersize=1)
    plt.plot(seg_index3, y3, 'b.', label='high emotion', markersize=1)

    plt.ylabel('Predicted Emotion Level')
    plt.xlabel('Sample Index')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('Regressor_results.png', dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # emotion_feature_gen()

    path = r'D:\SpectrumAnalysis\labelling_results'
    files = os.listdir(path)

    emotion_csv = [f for f in files if (f.endswith('dataset.xlsx') and f.startswith('emotion'))]

    df = pd.read_excel(os.path.join(path, emotion_csv[0]))
    print(df.head(3))

    # Feature Selection

    # features_name_lst = ['Volume', 'Decibel', 'Volume_var', 'Decibel_var', 'Volume_std', 'Decibel_std', 'Pause_percent']
    # features_selected_evaluation = []
    #
    # for i in range(1, len(features_name_lst)+1):
    #     for c in combinations(features_name_lst, i):
    #         combined = list((*c, 'Emotion'))
    #         rule = {'Volume': 0, 'Decibel': 1, 'Volume_var': 2, 'Decibel_var': 3, 'Volume_std': 4, 'Decibel_std': 5, 'Pause_percent': 6, 'Emotion': 7}
    #         sorted_features_name = sorted(combined, key=lambda x: rule[x])
    #         print(sorted_features_name)
    #         df_features = df.loc[:, sorted_features_name]
    #         features_selected_evaluation.append(regressor_evaluation(df_features))
    #
    #         # print(features_selected_evaluation)
    #
    # features_selected_evaluation = pd.DataFrame(features_selected_evaluation)
    # features_selected_evaluation = features_selected_evaluation.sort_values(features_selected_evaluation.columns[11]).T # sort by mape
    # features_selected_evaluation = features_selected_evaluation.values.tolist()
    #
    # fig = go.Figure(data=[go.Table(header=dict(values=['Volume', 'Decibel', 'Volume_var', 'Decibel_var', 'Volume_std', 'Decibel_std', 'Pause_percent', 'Bias', 'MAPE', 'MAE', 'RMSE', 'Error_sum']),
    #                                cells=dict(values=features_selected_evaluation))])
    #
    # fig.write_image("fig1.png")
    # fig.show()

    df_features = df.loc[:, ['Volume', 'Decibel', 'Decibel_var', 'Pause_percent', 'Emotion']]

    # retrieve emotion labels
    emotion_labels = np.array(df_features['Emotion'])

    # remove labels from features
    df_features = df_features.drop('Emotion', axis=1)

    # saving feature names for printing the variable importance
    feature_list = list(df_features.columns)

    # convert to numpy array
    features = np.array(df_features)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    predictions_lst = []
    labels_lst = []
    index_lst = []
    acc_classifier = []
    for train_index, test_index in kf.split(features, emotion_labels):
        train_features, train_labels = features[train_index], emotion_labels[train_index]
        test_features, test_labels = features[test_index], emotion_labels[test_index]
        rfc = RandomForestClassifier(n_estimators=1000, random_state=0)
        # rfc = BalancedRandomForestClassifier(n_estimators=1000) // downsampling the majority classes
        rfc.fit(train_features, train_labels)
        predictions = rfc.predict(test_features)

        predictions_lst = predictions_lst + list(predictions)
        labels_lst = labels_lst + list(test_labels)
        index_lst = index_lst + list(test_index)

        score_r = rfc.score(test_features, test_labels)
        acc_classifier.append(score_r)
        # print('Random Forest: ', score_r)

    print("5 fold average classification accuracy: ", np.mean(acc_classifier))

    draw_prediction_results(labels_lst, predictions_lst, index_lst)

    predictions_lst = []
    labels_lst = []
    pair_lst = []
    index_lst = []
    low_bias_lst = []
    neutral_bias_lst = []
    high_bias_lst = []
    all_bias_lst = []
    low_mae_lst = []
    neutral_mae_lst = []
    high_mae_lst = []
    all_mae_lst = []
    low_mape_lst = []
    neutral_mape_lst = []
    high_mape_lst = []
    all_mape_lst = []
    low_rmse_lst = []
    neutral_rmse_lst = []
    high_rmse_lst = []
    all_rmse_lst = []

    for train_index, test_index in kf.split(features, emotion_labels):
        train_features, train_labels = features[train_index], emotion_labels[train_index]
        test_features, test_labels = features[test_index], emotion_labels[test_index]

        rf = RandomForestRegressor(n_estimators=1000, random_state=0)
        rf.fit(train_features, train_labels)
        predictions = rf.predict(test_features)

        predictions_lst = predictions_lst + list(predictions)
        labels_lst = labels_lst + list(test_labels)
        index_lst = index_lst + list(test_index)
        pair_lst = pair_lst + list(zip(predictions, test_labels))
    print("Prediction results: ", pair_lst)

    draw_prediction_results(labels_lst, predictions_lst, index_lst)

    for pair in pair_lst:
        bias = pair[0] - pair[1]
        mae = abs(pair[0] - pair[1])
        mape = abs(pair[0] - pair[1]) / pair[1]
        rmse = np.square(bias)
        all_bias_lst.append(bias)
        all_mae_lst.append(mae)
        all_mape_lst.append(mape)
        all_rmse_lst.append(rmse)

        if pair[1] == 1:
            low_bias_lst.append(bias)
            low_mae_lst.append(mae)
            low_mape_lst.append(mape)
            low_rmse_lst.append(rmse)
        elif pair[1] == 2:
            neutral_bias_lst.append(bias)
            neutral_mae_lst.append(mae)
            neutral_mape_lst.append(mape)
            neutral_rmse_lst.append(rmse)
        else:
            high_bias_lst.append(bias)
            high_mae_lst.append(mae)
            high_mape_lst.append(mape)
            high_rmse_lst.append(rmse)

    low_bias = np.mean(low_bias_lst);
    neutral_bias = np.mean(neutral_bias_lst);
    high_bias = np.mean(high_bias_lst);
    whole_bias = np.mean(all_bias_lst)
    low_mae = np.mean(low_mae_lst);
    neutral_mae = np.mean(neutral_mae_lst);
    high_mae = np.mean(high_mae_lst);
    whole_mae = np.mean(all_mae_lst)
    low_mape = np.mean(low_mape_lst);
    neutral_mape = np.mean(neutral_mape_lst);
    high_mape = np.mean(high_mape_lst);
    whole_mape = np.mean(all_mape_lst)
    low_rmse = np.sqrt(np.mean(low_rmse_lst));
    neutral_rmse = np.sqrt(np.mean(neutral_rmse_lst));
    high_rmse = np.sqrt(np.mean(high_rmse_lst));
    whole_rmse = np.mean(all_rmse_lst)

    fig, ax = plt.subplots(1, 1)
    emotion_evaluate = [[low_bias, neutral_bias, high_bias, whole_bias],
                        [low_mape, neutral_mape, high_mape, whole_mape],
                        [low_mae, neutral_mae, high_mae, whole_mae],
                        [low_rmse, neutral_rmse, high_rmse, whole_rmse]]

    column_labels = ["Low", "Neutral", "High", "Overall"]
    df = pd.DataFrame(emotion_evaluate, columns=column_labels)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, rowLabels=["Bias", "MAPE", "MAE", "RMSE"], loc="center")
    plt.tight_layout()
    plt.savefig('Regressor_evaluation.png', dpi=500, bbox_inches='tight')
    plt.show()

    # Train another random forest regressor
    rf = RandomForestRegressor(n_estimators=1000, random_state=0)

    # Instantiate model with 1000 decision trees
    train_features, test_features, train_labels, test_labels = train_test_split(features, emotion_labels, test_size=0.2,
                                                                                random_state=0)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    compare = list(zip(predictions, test_labels))
    errors = abs(predictions - test_labels)
    mape = np.mean(100 * (errors / test_labels))
    accuracy = 100 - mape
    print('MAPE accuracy:', round(accuracy, 2), "%.")

    # Get numerical feature importance
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    draw_variable_importance(feature_list, importances)
