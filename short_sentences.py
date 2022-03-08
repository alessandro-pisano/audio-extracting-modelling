import librosa
import numpy as np
import os
import soundfile
import time
import wave

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

class AudioSentence(): 
    def __init__(self, file, one_channel=False):
        self.file = file
        self.one_channel = one_channel
    
    def save_wav_channel(self, fn, wav, channel, nch):
        """
        Take Wave_read object as an input and save one of its
        channels into a separate .wav file.
        """
        # Read data
        depth = wav.getsampwidth()
        wav.setpos(0)
        sdata = wav.readframes(wav.getnframes())

        # Extract channel data (24-bit data not supported)
        typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
        if not typ:
            print(f"Sample width {depth} not supported")

        data = np.frombuffer(sdata, dtype=typ)
        ch_data = data[channel::nch]

        # Save channel to a separate file
        outwav = wave.open(fn, 'w')
        outwav.setparams(wav.getparams())
        outwav.setnchannels(1)
        outwav.writeframes(ch_data.tobytes())
        outwav.close()

    def resampling_wav(self, file_path):
        """ Resampling audio files """
        data, samplerate = soundfile.read(file_path)
        soundfile.write(file_path, data, samplerate, subtype='PCM_16')

    def extract_audio(self):
        """ Extracting audio voices"""
        dir_f = os.getcwd() + "/" + "Audio_Split"
        os.makedirs(dir_f, exist_ok=True)
        if self.file.endswith(".wav"):
            new_name_A = dir_f + "/" + self.file.split("/")[-1][:-4] + "_A.wav"
            new_name_B = dir_f + "/" + self.file.split("/")[-1][:-4] + "_B.wav"
            try:
                wav = wave.open(self.file)    
            except:
                print("Resampling", self.file)
                self.resampling_wav(self.file)
                wav = wave.open(self.file)
            # Getting number of channels
            nch   = wav.getnchannels()
            if 1 >= nch:
                #print(f"{self.file}: cannot extract channel {2} out of {nch}. Extract only one channel")
                self.one_channel = True
                new_name_ = dir_f + "/" + self.file.split("/")[-1][:-4] + "_one_channel.wav"
                self.save_wav_channel(new_name_, wav, 0, nch)
            else:
                self.save_wav_channel(new_name_A, wav, 0, nch)
                self.save_wav_channel(new_name_B, wav, 1, nch)
        else:        
            print(f"Error with file {self.file}. Format not supported.")

    def match_target_amplitude(self, sound, target_dBFS):
        """Adjust target amplitude"""
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def get_sentences(self, audio_seg, norm=-20.0, min_silence=1350, silence=-40):
        """Getting sentences"""
        # Normalize audio_segment to -20dBFS 
        normalized_sound = self.match_target_amplitude(audio_seg, norm)
        # Speaking chunks
        nonsilent_data = detect_nonsilent(normalized_sound, 
                                          min_silence_len=min_silence, 
                                          silence_thresh=silence, 
                                          seek_step=1)    
        return nonsilent_data

    def start_ending(self, sentences, audio_track, audio_path, track, min_silence):
        count = 0
        for start, end in sentences:
            if end-start>15000 and min_silence > 500:
                min_silence = min_silence-100
                sentences_cut = self.get_sentences(audio_track[start:end], 
                                                   min_silence=min_silence, 
                                                   silence=-30)
                sentences_cut = (np.array(sentences_cut) + start).tolist()
                self.start_ending(sentences_cut, 
                                  audio_track, 
                                  audio_path, 
                                  track, 
                                  min_silence)
            elif end-start>=1350:
                new_sent = audio_track[start:end]
                path = audio_path + self.file.split("/")[-1][:-4] + "_" + str(start) + "_" + str(end) + track + ".wav"
                new_sent.export(path, format="wav")
            else:
                #print(audio_path, "A: less than 2 sec:", start, end)
                count +=1

        return count

    def save_speech(self):
        """Saving features"""
        audio_path = "Audio_Split/" + self.file.split("/")[-1][:-4] + "/"
        os.makedirs(audio_path, exist_ok=True)

        if self.one_channel:
            audio_ch_path = "Audio_Split/" + self.file.split("/")[-1][:-4] + "_one_channel.wav"
            audio_ch = AudioSegment.from_wav(audio_ch_path)
            sentences_ch = self.get_sentences(audio_ch)
            count_ch = self.start_ending(sentences_ch, audio_ch, audio_path, "_one_channel", 800)
            os.remove(audio_ch_path)
            print(f"Done with {self.file}")
            return count_ch
        else:    
            audio_A_path = "Audio_Split/" + self.file.split("/")[-1][:-4] + "_A.wav" 
            audio_B_path = "Audio_Split/" + self.file.split("/")[-1][:-4] + "_B.wav" 
            audio_A = AudioSegment.from_wav(audio_A_path)
            audio_B = AudioSegment.from_wav(audio_B_path) 
            sentences_A = self.get_sentences(audio_A)
            sentences_B = self.get_sentences(audio_B)
            count_A = self.start_ending(sentences_A, audio_A, audio_path, "_A", 1350)
            os.remove(audio_A_path)
            count_B = self.start_ending(sentences_B, audio_B, audio_path, "_B", 1350)
            os.remove(audio_B_path)
            print(f"Done with {self.file}")
            return [count_A, count_B]

    def run(self):
        self.extract_audio()
        self.save_speech()