import librosa
import matplotlib.pyplot as plt
import numpy as np
import os


class OverlapFeaturesGenerator:

    def __init__(self, wl, hl, sr=16000):
        """
        :param wl: windows length in milliseconds to sample one spectrogram data point in stft
        :param hl: hop length in milliseconds to slide the window
        :param sr:
        """

        self.sr = sr
        self.window_length = int(sr * (wl / 1000))
        self.hop_length = int(sr * (hl / 1000))

    def get_attributes(self):
        """
        :return: return sampling length in frames, hop length in frames, sampling rate
        """

        return self.window_length, self.hop_length, self.sr

    def generate_mels(self, wav_file_path, n_mels=128):
        """
        :param wav_file_path: path of audio file relative to project dir
        :param n_mels: number of mel frequency bins
        :return: log power spectrogram relative to max power, normalized log power spectrogram
        """

        y, sr = librosa.load(wav_file_path, sr=None)
        S = librosa.feature.melspectrogram(y, sr, hop_length=self.hop_length, n_fft=self.window_length, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db_norm = self.normalize_matrix(S_db)

        return S_db, S_db_norm

    def generate_zcr(self, wav_file_path):
        """
        :param wav_file_path: path of audio file relative to project dir
        :return: zero crossing array with shape (1, n), [0][i] indicates i th window's zero crossing rate
        """

        y, sr = librosa.load(wav_file_path, sr=None)
        arr_zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.window_length, hop_length=self.hop_length)

        return arr_zcr

    @staticmethod
    def normalize_matrix(m):
        """
        :param m: 2D matrix
        :return: normalized 2D matrix
        """

        max_val = np.max(m)
        min_val = np.min(m)
        diff = max_val - min_val
        temp = np.empty_like(m)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                temp[i][j] = (m[i][j] - min_val) / diff
        return temp

    def generate_images(self, wav_file_path, file_name):
        """
        :param wav_file_path: audio file to create and save zcr-enhanced image, mel-grayscale image, viridis image, and zcr plot
        :param file_name: image name for storage
        :return: None
        """

        # create directories
        if not os.path.isdir("./mel_spectrum_viridis"):
            os.mkdir("./mel_spectrum_viridis")

        if not os.path.isdir("./mel_spectrum_gray"):
            os.mkdir("./mel_spectrum_gray")

        if not os.path.isdir("./mel_spectrum_zcr"):
            os.mkdir("./mel_spectrum_zcr")

        arr_zcr = self.generate_zcr(wav_file_path)
        # plt.imsave("./zcr_plot/" + file_name, arr_zcr[0])

        _, norm_log_power_mel_spectrum = self.generate_mels(wav_file_path)
        plt.imsave('./mel_spectrum_viridis/' + file_name, norm_log_power_mel_spectrum, origin="lower", cmap="viridis")
        plt.imsave('./mel_spectrum_gray/' + file_name, norm_log_power_mel_spectrum, origin="lower", cmap="gray")

        zcr_enhanced_image = np.empty((norm_log_power_mel_spectrum.shape[0], norm_log_power_mel_spectrum.shape[1], 3))
        for i in range(norm_log_power_mel_spectrum.shape[0]):
            for j in range(norm_log_power_mel_spectrum.shape[1]):
                zcr_enhanced_image[i][j][0] = arr_zcr[0][j]
                zcr_enhanced_image[i][j][1] = 1 - norm_log_power_mel_spectrum[i][j]
                zcr_enhanced_image[i][j][2] = 1 - norm_log_power_mel_spectrum[i][j]
        plt.imsave('./mel_spectrum_zcr/' + file_name, zcr_enhanced_image, origin="lower")
        plt.close('all')

    def generate_test_images(self, wav_file_path, file_name):
        """
        :param wav_file_path: audio file to create and save zcr-enhanced image, mel-grayscale image, viridis image, and zcr plot
        :param file_name: image name for storage
        :return: None
        """

        if not os.path.isdir("./mel_spectrum_viridis_test"):
            os.mkdir("./mel_spectrum_viridis_test")

        if not os.path.isdir("./mel_spectrum_gray_test"):
            os.mkdir("./mel_spectrum_gray_test")

        if not os.path.isdir("./mel_spectrum_zcr_test"):
            os.mkdir("./mel_spectrum_zcr_test")

        arr_zcr = self.generate_zcr(wav_file_path)

        _, norm_log_power_mel_spectrum = self.generate_mels(wav_file_path)
        plt.imsave('./mel_spectrum_viridis_test/' + file_name, norm_log_power_mel_spectrum, origin="lower",
                   cmap="viridis")
        plt.imsave('./mel_spectrum_gray_test/' + file_name, norm_log_power_mel_spectrum, origin="lower", cmap="gray")

        zcr_enhanced_image = np.empty((norm_log_power_mel_spectrum.shape[0], norm_log_power_mel_spectrum.shape[1], 3))
        for i in range(norm_log_power_mel_spectrum.shape[0]):
            for j in range(norm_log_power_mel_spectrum.shape[1]):
                zcr_enhanced_image[i][j][0] = arr_zcr[0][j]
                zcr_enhanced_image[i][j][1] = 1 - norm_log_power_mel_spectrum[i][j]
                zcr_enhanced_image[i][j][2] = 1 - norm_log_power_mel_spectrum[i][j]
        plt.imsave('./mel_spectrum_zcr_test/' + file_name, zcr_enhanced_image, origin="lower")
        plt.close('all')


if __name__ == '__main__':
    ######################################## Overalp Features Generating ###############################################

    # # Feature Generator with windows length = 25ms, hop length = 10ms, given default sampling rate=16000
    ofg = OverlapFeaturesGenerator(wl=25, hl=10)
    # print(ofg.get_attributes())
    #
    # src_dir = './segments_mono_10'
    # # session_dir_lst = ['S02_audio_MONO', 'S03_audio_MONO', 'S09_audio_MONO', 'S13_audio_MONO', 'S20_audio_MONO']
    # session_dir_lst = ['S22_audio_MONO']
    #
    #
    # for i in session_dir_lst:
    #     session_dir = os.path.join(src_dir, i)
    #     audio_files = os.listdir(session_dir)
    #     audio_files_path = [session_dir + "\\" + f for f in audio_files]
    #
    #     for wav_path in audio_files_path:
    #         file_name = os.path.basename(wav_path)[:-4]+'.png'
    #         print(file_name)
    #         # ofg.generate_images(wav_path, file_name)
    #         ofg.generate_test_images(wav_path, file_name)

    ######################################## Overalp Features Generating ###############################################

    ####################################### Plot ZCR Colormap ##########################################################
    # cp1 = np.linspace(0, 1, 200)
    # cp2 = np.linspace(1, 0, 200)
    # red, green = np.meshgrid(cp1, cp2)
    # blue = green
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # colormap = np.dstack((red, green, blue))
    # plt.imshow(colormap, origin="lower", extent=[0, 2, -1, 0])
    # ax.annotate('', xy=(0.67, -0.05), xycoords='axes fraction', xytext=(1, -0.05),
    #             arrowprops=dict(arrowstyle="<-", color='black'))
    # ax.annotate('', xy=(-0.025, 0.85), xycoords='axes fraction', xytext=(-0.025, 1.1),
    #             arrowprops=dict(arrowstyle="<-", color='black'))
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])
    # plt.xlabel("Zero Crossing Rate")
    # plt.ylabel("Log Power Mel Spectrum")
    # plt.title("Color map", fontsize=10)
    # plt.savefig('zcr_colormap.png', dpi=500, bbox_inches='tight')
    # plt.show()

    test_audio = 'D:/SpectrumAnalysis/segments_mono_10/S02_audio_MONO/S02_audio_MONO_16_16000_split.wav'
    arr_zcr = ofg.generate_zcr(test_audio)
    _, norm_log_power_mel_spectrum = ofg.generate_mels(test_audio)
    zcr_enhanced_image = np.empty((norm_log_power_mel_spectrum.shape[0], norm_log_power_mel_spectrum.shape[1], 3))
    melfb = librosa.mel_frequencies(n_mels=128, fmax=8000)

    print(melfb.shape)
    print(melfb)

    for i in range(norm_log_power_mel_spectrum.shape[0]):
        for j in range(norm_log_power_mel_spectrum.shape[1]):
            zcr_enhanced_image[i][j][0] = arr_zcr[0][j]
            zcr_enhanced_image[i][j][1] = 1 - norm_log_power_mel_spectrum[i][j]
            zcr_enhanced_image[i][j][2] = 1 - norm_log_power_mel_spectrum[i][j]


    plt.plot(np.arange(len(arr_zcr[0]))*0.01, arr_zcr[0])
    plt.xlabel('time')
    plt.ylabel('zero crossing rate')
    plt.savefig('zcr_plot.png', dpi=500, bbox_inches='tight')
    plt.show()

    plt.imshow(zcr_enhanced_image, origin='lower')
    plt.xticks([0, 20, 40, 60, 80, 100, 120, 140], labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    plt.yticks([0, 15, 31, 47, 63, 79, 95, 111, 127], labels=[int(melfb[0]), int(melfb[15]), int(melfb[31]), int(melfb[47]), int(melfb[63]), int(melfb[79]), int(melfb[95]), int(melfb[111]), int(melfb[127])])
    plt.xlabel('time')
    plt.ylabel('128 mels frequency (Hz)')
    plt.savefig('zcr_enhanced.png', dpi=500, bbox_inches='tight')
    plt.show()




