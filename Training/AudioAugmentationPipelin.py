import random
import torchaudio.transforms as T
import torch
import librosa as lb
import numpy as np

from tqdm import tqdm
from Training.config import AugmentConfig
from typing import Optional

class AudioAugmentationPipeline:
 
    def __init__(
                self,
                sr: int = 15_872,  # Input audio sample rate in Hz
                n_mels: int = 62, # Number of mel bands
                hop: int = 512, # Hop size for STFT
                n_fft: int = 512, # Number of FFT points
                config: Optional[AugmentConfig] = None,
                seed: Optional[int] = None,
                background_files: Optional[list[str]] = None
                ):
        
        self.background_files = background_files
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop = hop
        self.hanningWindow = np.hanning(sr)

        self.config    = config or AugmentConfig()

        # Instances for calculating log mel spectograms
        self.mel_spectogram = T.MelSpectrogram(sample_rate=self.sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
        self.amplitude_to_db = T.AmplitudeToDB(top_db=90)
 
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # This is the main method that applies the augmentation pipeline to an input audio array. 
    def _add_noise(
        self, audio: np.ndarray, snr_db: float, noise_type: str
    ) -> np.ndarray:

        signal_power = np.mean(audio ** 2) + 1e-9
 
        if noise_type == "white":
            noise = np.random.randn(len(audio))
 
        elif noise_type == "pink":
            f = np.fft.rfftfreq(len(audio))
            f[0] = 1.0          # avoid divide-by-zero at DC
            spectrum = np.random.randn(len(f)) / np.sqrt(f)
            noise = np.fft.irfft(spectrum, n=len(audio))
 
        else:
            return audio
 
        noise_power = np.mean(noise ** 2) + 1e-9
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise *= np.sqrt(target_noise_power / noise_power)
        return audio + noise

    # Pitch shifting using librosa's pitch_shift function, which changes the pitch of the audio by a specified number of semitones.
    def _pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        return lb.effects.pitch_shift(
            audio.astype(np.float32),
            sr=self.sr,
            n_steps=n_steps,
        ).astype(np.float64)
 
    # Volume scaling by multiplying the audio signal by a random gain factor within a specified range, which simulates different recording volumes or distances from the microphone.
    def _volume_scale(self, audio: np.ndarray, gain: float) -> np.ndarray:
        return np.clip(audio * gain, -1.0, 1.0)
    
    # Time shifting by rolling the audio array by a random number of samples, which simulates the effect of the sound occurring slightly earlier or later in time.
    def _time_shift(self, audio: np.ndarray, shift: float) -> np.ndarray:
        # Shift is a fraction of total length, e.g. (-0.2, 0.2)
        n = int(shift * len(audio))
        return np.roll(audio, n) 
 
    # Feature-level augmentation that applies SpecAugment by randomly masking out certain frequency bands and time frames in the log-Mel spectrogram, which helps to improve the model's robustness to variations in the input features. 
    def _spec_augment(self, result: np.ndarray) -> np.ndarray:
        result = result.copy()
        n_mels, n_frames = result.shape
        cfg = self.config
        fill = result.mean()
        for _ in range(np.random.randint(cfg.n_freq_masks)):
            f  = random.randint(0, n_mels-1)
            result[f, :] = fill          # freq masking is fine for claps

        for _ in range(np.random.randint(cfg.n_time_masks)):
            t  = random.randint(0, n_frames-1)
            result[:, t] = fill

        return result
    
    # This is the main method that gets called with the raw audio data. It applies the various augmentations based on the probabilities defined in the AugmentConfig, extracts the log-Mel spectrogram, and returns it as a numpy array of type float32 for use in training the CNN classifier.
    def process(self, audio: np.ndarray, noise=True, pitch=True, volume=True, spec_aug=True, time_shift=True) -> np.ndarray:
        cfg   = self.config
        audio = audio.flatten().astype(np.float64)
 
        # 1. Noise
        if random.random() < cfg.noise_prob and noise:
            pool = list(cfg.noise_types)
            audio = self._add_noise(audio, random.uniform(*cfg.noise_snr_range), random.choice(pool))
 
        # 3. Pitch shift
        if random.random() < cfg.pitch_shift_prob and pitch:
            audio = self._pitch_shift(audio, random.uniform(*cfg.pitch_shift_range))

        # 4. Time shift
        if random.random() < cfg.time_shift_prob and time_shift:
            audio = self._time_shift(audio, random.uniform(*cfg.time_shift_range))
 
        # 5. Volume scaling
        if random.random() < cfg.volume_scale_prob and volume:
            audio = self._volume_scale(audio, random.uniform(*cfg.volume_gain_range))

        audio /= (np.max(np.abs(audio)) + 1e-9)
        
        # 6. Extract log-Mel spectrogram
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        mfcc = self.mel_spectogram(audio_tensor)
        mfcc = self.amplitude_to_db(mfcc).numpy()

        # 7. SpecAugment
        if random.random() < cfg.spec_augment_prob and spec_aug:
            mfcc = self._spec_augment(mfcc)
 
        return mfcc
 
