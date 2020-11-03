import numpy as np
import musdb
import librosa
import random
import keras
from time import time

class generator(keras.utils.Sequence):
    def __init__(self, steps_per_epoch, sbsts, splt, source_name, batch_size, tracks_in_batch, sampling_rate, win_len, hop_len, sample_len):
        self.steps_per_epoch = steps_per_epoch
        self.tracks_in_batch = tracks_in_batch
        self.sbsts = sbsts
        self.splt = splt
        self.source_name = source_name
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.win_len = win_len
        self.hop_len = hop_len
        self.sample_len = sample_len
        self.mus = musdb.DB(root='./musdb18', subsets=sbsts, split=splt)
        self.track_number = np.arange(len(self.mus))
        np.random.shuffle(self.track_number)
        self.cur_index = 0
        self.duration = (self.hop_len/self.sampling_rate)*((self.batch_size/self.tracks_in_batch) + (self.sample_len-1)) + (self.win_len-self.hop_len)/self.sampling_rate

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        mixes = []
        targets = []
        for i in range(self.tracks_in_batch):
            orig_sr, mix, target = self.get_random_track_piece() 
            mix_power_spec = self.get_power_spec(mix,
                                                 orig_sr,
                                                 self.sampling_rate,
                                                 self.win_len,
                                                 self.hop_len)
            target_power_spec = self.get_power_spec(target,
                                                    orig_sr,
                                                    self.sampling_rate,
                                                    self.win_len,
                                                    self.hop_len)
            mix_portion, target_portion = self.split_and_prep(mix_power_spec, target_power_spec) 
            mixes = mixes + mix_portion
            targets = targets + target_portion   
        # Преобразование списков в массивы
        mix_batch = np.array(mixes)
        target_batch = np.array(targets)
        # Добавление старшего измерения, соответствующего каналу
        mix_batch = np.expand_dims(mix_batch, len(mix_batch.shape))         
        # Возврат батча в вызывающую функцию
        return mix_batch, target_batch

    def get_random_track_piece(self):
        random.seed(int(time()%1*1000000))
        if self.cur_index == self.track_number.shape[0]:
            np.random.shuffle(self.track_number)
            self.cur_index = 0
        track = self.mus[self.track_number[self.cur_index]]
        self.cur_index += 1
        track.chunk_duration = self.duration
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
        mix = track.audio.T
        target = track.targets[self.source_name].audio.T
        channel = random.randint(0, mix.shape[0]-1)
        return track.rate, mix[channel], target[channel]
    
    def get_power_spec(self, audio, orig_sr, new_sr, win_len, hop_len):
        if orig_sr!= new_sr:
            audio = librosa.core.resample(audio, orig_sr, new_sr)
        stft_ar = librosa.stft(audio, win_len, hop_len, center = False)
        return np.abs(stft_ar)
    
    def split_and_prep(self, mix_power_spec, target_power_spec):      
        mix_portion = []
        target_portion = []
        for i in range(mix_power_spec.shape[1]-self.sample_len+1):
            mix_sample = np.copy(mix_power_spec[:,i:i+self.sample_len])
            target_sample = np.copy(target_power_spec[:,i+self.sample_len//2])
            target_sample = self.target_sample_to_mask(target_sample, mix_sample)
            target_portion.append(target_sample)
            mix_sample = self.normalize_pow_spec(mix_sample)
            mix_portion.append(mix_sample)
        return mix_portion, target_portion
                    
    def target_sample_to_mask(self, target_sample, mix_sample):
        for f in range(target_sample.shape[0]):
            if mix_sample[f, self.sample_len//2] != 0:
                target_sample[f] /= mix_sample[f,self.sample_len//2]
                if target_sample[f] > 0.5:
                    target_sample[f] = 1
                else:
                    target_sample[f] = 0
        return target_sample
                    
    def normalize_pow_spec(self, pow_spec):
        max = np.max(pow_spec)
        if max != 0:
            pow_spec /= max
        return pow_spec 
    
    