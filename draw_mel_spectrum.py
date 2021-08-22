import numpy as np
from pydub import AudioSegment
import pydub
import os
import wave
import json
from matplotlib import pyplot as plt
import librosa
import librosa.display


def draw_all():
    filename = './wav_mono/S03_audio_MONO.wav'
    y, sr = librosa.load(filename)

    # trim silent edges
    wav_data, _ = librosa.effects.trim(y)
    # librosa.display.waveplot(wav_data, sr=sr)
    # plt.ylabel('Amplitude')
    # plt.show()

    n_fft = 2048
    d = np.abs(librosa.stft(wav_data[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))
    # plt.plot(d)
    # plt.show()

    hop_length = 512
    d = np.abs(librosa.stft(wav_data, n_fft=n_fft, hop_length=hop_length))
    # librosa.display.specshow(d, sr=sr, x_axis='time', y_axis='linear')
    # plt.colorbar()
    # plt.show()

    db = librosa.amplitude_to_db(d, ref=np.max)
    # librosa.display.specshow(db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()

    n_mels = 128
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    #
    # plt.figure(figsize=(15,4))
    # plt.subplot(1, 3, 1)
    # librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='linear')
    # plt.ylabel('Mel filter')
    # plt.colorbar()
    # plt.title('1. Filter bank for converting from Hz to mels.')
    #
    # plt.subplot(1, 3, 2)
    # mel_10 = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=10)
    # librosa.display.specshow(mel_10, sr=sr, hop_length=hop_length, x_axis='linear')
    # plt.ylabel('Mel filter')
    # plt.colorbar()
    # plt.title('2. 10 mel filter bank')
    #
    # plt.subplot(1, 3, 3)
    # idxs_to_plot = [0, 9, 49, 99, 127]
    # for i in idxs_to_plot:
    #     plt.plot(mel[i])
    # plt.legend(labels=[f'{i+1}' for i in idxs_to_plot])
    # plt.title('3. triangular filters separately')
    # plt.tight_layout()
    #
    # plt.show()

    s = librosa.feature.melspectrogram(wav_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    s_db = librosa.power_to_db(s, ref=np.max)
    librosa.display.specshow(s_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB');
    plt.show()


def draw_mel_spectrum():
    src_dir = './segments_mono_10'
    session_dir_lst = ['S02_audio_MONO', 'S03_audio_MONO', 'S09_audio_MONO', 'S20_audio_MONO', 'S13_audio_MONO']
    for i in session_dir_lst:
        print(i)
        session_dir = os.path.join(src_dir, i)
        files_name = os.listdir(session_dir)
        files_path = [session_dir + "\\" + f for f in files_name]

        for file in files_path:
            y, sr = librosa.load(file, sr=None)
            # print(len(y), sr)
            s = librosa.feature.melspectrogram(y, sr, hop_length=int(0.010 * sr), n_fft=int(0.025 * sr), n_mels=128)
            s_db = librosa.power_to_db(s, ref=np.max)
            # fig = plt.figure()
            # plt.margins(0, 0)
            # librosa.display.specshow(s_db)
            # librosa.display.specshow(s_db, sr=sr, x_axis='time', y_axis='mel', hop_length=int(0.010*sr))
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('Mel Spectrum')
            plt.tight_layout()
            plt.imsave('./mel_spectrum/' + os.path.basename(file)[:-4] + '.png', s_db)
            plt.close('all')


def read_wav(wav_path):
    """
    Read wac file for mono channel. Return json.
    :param wav_path: file path of WAV
    """
    wav_file = wave.open(wav_path, 'r')
    numchannel = wav_file.getnchannels()  # channels number
    samplewidth = wav_file.getsampwidth()  # stride
    framerate = wav_file.getframerate()  # sampling rate
    numframes = wav_file.getnframes()  # samples count
    print("channel", numchannel)
    print("sample_width", samplewidth)
    print("framerate", framerate)
    print("numframes", numframes)
    wav_data = wav_file.readframes(numframes)
    wav_data = np.fromstring(wav_data, dtype=np.int16)
    wav_data = wav_data * 1.0 / (max(abs(wav_data)))  # normalization
    # generate the audio data, since ndarray could not been convert to JSON, firstly convert it to list, then store as JSON.
    dictionary = {"channel": numchannel,
                  "samplewidth": samplewidth,
                  "framerate": framerate,
                  "numframes": numframes,
                  "WaveData": list(wav_data)}
    return json.dumps(dictionary)


def draw_spectrum(wav_data, framerate):
    """
    Draw Spectrum
    :param wav_data: audio data
    :param framerate: sampling rate
    """
    time = np.linspace(0, len(wav_data) / framerate * 1.0, num=len(wav_data))
    plt.figure(1)
    plt.plot(time, wav_data)
    plt.grid(True)
    plt.show()
    plt.figure(2)
    pxx, freqs, bins, im = plt.specgram(wav_data, NFFT=1024, Fs=16000, noverlap=900)
    plt.show()
    print(pxx)
    print(freqs)
    print(bins)
    print(im)


def run_main():
    """
        main function
    """
    path = './wav_file'
    paths = os.listdir(path)
    wav_paths = []
    for wav_path in paths:
        wav_paths.append(path + '/' + wav_path)

    print(wav_paths)

    # read wav file
    for wav_path in wav_paths:
        wav_json = read_wav(wav_path)
        print(wav_json)
        wav = json.loads(wav_json)
        wav_data = np.array(wav['WaveData'])
        framerate = int(wav['framerate'])
        draw_spectrum(wav_data, framerate)


if __name__ == '__main__':
    # draw_mel_spectrum()
    sr = 16000
    n_fft = 512
    n_mels = 26
    mel = librosa.filters.mel(sr=16000, n_fft=512, n_mels=26)
    plt.subplot(1, 2, 1)
    librosa.display.specshow(mel, sr=16000, hop_length=160, x_axis='linear')
    plt.ylabel('Mel Filter')
    plt.colorbar()
    # plt.savefig('mel_filters_bank.png', dpi=500, bbox_inches='tight')

    # plt.show()

    plt.subplot(1, 2, 2)
    f = np.linspace(0, sr/2, int((n_fft/2)+1))
    for i in range(n_mels):
        plt.plot(f, mel[i])
    plt.ylabel('Mel Filter Coefficient')
    plt.xlabel('Hz')
    # plt.savefig('mel_triangular_filters.png', dpi=500, bbox_inches='tight')
    plt.show()

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))  # Convert Hz to Mel

    print(high_freq_mel)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)  # Equally spaced in Mel scale
    print(mel_points)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    print(hz_points)
    bin = np.floor((512 + 1) * hz_points / sr)



