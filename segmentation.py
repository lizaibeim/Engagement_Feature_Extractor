import os
import wave
import numpy as np
import librosa
from ffmpy3 import FFmpeg


def split(src_dir, dst_dir, win_time_stride, step_time):
    """
    :param src_dir: directory of the clips to cut
    :param dst_dir: directory to store the new cut clips
    :param win_time_stride: the time duration of the new cut clip
    :param step_time: difference time between two cut clip, overlap_time = win_time - step_time
    :return:
    """
    files = os.listdir(src_dir)
    files = [src_dir + "\\" + f for f in files if f.endswith('.wav')]

    for i in range(len(files)):
        filename = files[i]
        f = wave.open(filename, 'rb')

        params = f.getparams()
        # Get digital parameters of audio file, channels number, sampling width(bits per second), sampling rate,
        # sampling frames
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
        f.close()

        wave_data = np.frombuffer(str_data, dtype=np.short)

        if nchannels > 1:
            wave_data.shape = -1, 2
            wave_data = wave_data.T
            temp_data = wave_data.T
        else:
            wave_data = wave_data.T
            temp_data = wave_data.T

        win_num_frames = int(framerate * win_time_stride)
        step_num_frames = int(framerate * step_time)

        print("window frames: ", win_num_frames, "step frames: ", step_num_frames)
        cut_num = int(((nframes - win_num_frames) / step_num_frames) + 1)  # how many segments for one wav file
        step_total_num_frames = 0

        for j in range(cut_num):
            file_abs_path = os.path.splitext(os.path.split(filename)[-1])[0]
            file_save_path = os.path.join(dst_dir, file_abs_path)

            if not os.path.exists(file_save_path):
                os.makedirs(file_save_path)

            out_file = os.path.join(file_save_path, os.path.splitext(os.path.split(filename)[-1])[0] + '_%d_%s_split.wav' % (j, framerate))
            start = step_num_frames*j
            end = step_num_frames*j + win_num_frames
            temp_data_temp = temp_data[start:end]
            step_total_num_frames = (j+1)*step_num_frames
            temp_data_temp.shape = 1, -1
            temp_data_temp = temp_data_temp.astype(np.short)
            f = wave.open(out_file, 'wb')
            f.setnchannels(nchannels)
            f.setsampwidth(sampwidth)
            f.setframerate(framerate)
            f.writeframes(temp_data_temp.tostring())
            f.close()

        print("Total number of frames :", nframes, " Extract frames: ", step_total_num_frames)


def other_to_wav_float32(src_dir, dst_dir):
    """
    convert other format of audio file to wav format with float32 data type, single channel, 16000Hz sampling rate
    :return:
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        for name in files:
            audio_name_src = os.path.join(root, name)
            audio_name_dst = os.path.join(dst_dir, name.split('.')[0][:-3]+'MONO.wav')
            ff = FFmpeg(
                executable='C:\\PATH_Programs\\ffmpeg.exe',
                inputs={audio_name_src: None},
                outputs={audio_name_dst: ['-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000']}
            )
            ff.run()


if __name__ == "__main__":
    # other_to_wav_float32('./wav_stereo', './wav_mono')
    # f = wave.open('./wav_mono/S02_audio_MONO.wav', 'rb')
    #
    # y, sr = librosa.load('./wav_mono/S02_audio_MONO.wav')
    # print(y, sr)

    src_dir0 = './wav_stereo'
    src_dir2 = './wav_mono'
    dst_dir1 = './segments_stereo_10'
    dst_dir2 = './segments_mono_10'
    dst_dir3 = './segments_stereo_30'
    dst_dir4 = './segments_mono_30'

    # split(src_dir0, dst_dir1, 10)
    split(src_dir2, dst_dir2, 1.5, 1)
    # split(src_dir0, dst_dir3, 30)
    # split(src_dir2, dst_dir4, 30)







