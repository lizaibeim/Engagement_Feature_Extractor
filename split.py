import os
import struct
import wave

import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm
from pydub import AudioSegment



def run():
    path = './segments/S02_audio_STE/S02_audio_STE_0_48000_split.wav'
    path2 = './wav_file/M001_S02_audio_M_C.wav'
    f = wave.open(path2, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print(nchannels, sampwidth, framerate, nframes)

class voice_split(object):
    def __init__(self, in_path, out_path):
        # input path of the wav file
        self.in_path = in_path
        # output path of the wav file
        self.out_path = out_path
        # time interval for split (/s)
        self.voiceIntervalToSplit = 30

    # split the file, store separately
    def split_save(self):
        sig, self.sample_rate = sf.read(self.in_path)
        print('Current Read File:%s' % self.in_path)
        print('Sampling rate: %d' % self.sample_rate)
        print('Time duration: %s' % (sig.shape[0] / self.sample_rate), 'seconds')

        dd = { }

        x = [i / self.sample_rate for i in range(sig.shape[0])]
        y = list(dd.values())

        # split file by time interval
        voiceTimeStamp = list(dd.keys())
        voiceSegmentsTimeList = []
        OneSegmentTimeList = []
        start = -1

        for k, v in enumerate(voiceTimeStamp):
            if len(OneSegmentTimeList) == 0:
                start = v
            OneSegmentTimeList.append(v)
            if v - start > self. voiceIntervalToSplit * self.sample_rate:
                voiceSegmentsTimeList.append(OneSegmentTimeList)
                OneSegmentTimeList = []

        # if audio file smaller than time interval
        if len(voiceSegmentsTimeList) == 0:
            voiceSegmentsTimeList.append(OneSegmentTimeList)

        for segmentTimeList in voiceSegmentsTimeList:
            voiceTimeStamp1 = int(max(0, segmentTimeList[0] - 0.8 * self.sample_rate))
            voiceTimeStamp2 = int(min(sig.shape[0], segmentTimeList[-1] + 0.8 * self.sample_rate))
            self.wav_write_by_interval(wav_path=self.in_path, out_data=sig, start_time=voiceTimeStamp1, end_time=voiceTimeStamp2)


    # wav file write out by intervals
    def wav_write_by_interval(self, wav_path, out_data, start_time, end_time):
        out_data = out_data[start_time:end_time]
        file_abs_path = os.path.splitext(os.path.split(wav_path)[-1])[0]
        file_save_path = os.path.join(self.out_path, file_abs_path)

        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)

        out_file = os.path.join(file_save_path, os.path.splitext(os.path.split(wav_path)[-1])[0] + '_%d_%d_%s_split.wav' % (start_time, end_time, self.sample_rate))

        # check file exist or not
        if not os.path.exists(out_file):
            print('generating file:' , out_file)
            with wave.open(out_file, 'wb') as out_wave:
                nchannels = 1
                sampwidth = 2
                fs = 8000
                data_size = len(out_data)
                framerate = int(fs)
                nframes = data_size
                comptype = "NONE"
                compname = "not compressed"
                out_wave.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
                for v in out_data:
                    out_wave.writeframes(struct.pack('h', int(v * 64000 / 2)))

if __name__ == '__main__':
    run()























