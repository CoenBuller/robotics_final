import os
import numpy as np
from PitchExtraction import FastMelSpec

SAMPLE_RATE = 15_872 # Input audio sample rate in Hz
WINDOW_DURATION = 1 # Audio window duration in seconds
CHUNK_DURATION = 0.25 # Audio chunk duration in seconds
N_FFT = 512 # Number of FFT points
N_MELS = 62 # Number of mel bands
HOP = 512 # Hop size for STFT
MEL_FB_PATH = "mel_fb.npy" # File containing the pre-computed mel filterbank
NOISE_FILE = "average_motor_noise.npy" # File containing the pre-computed motor noise profile for spectral subtraction.

class AudioProcessor:
    def __init__(
        self,
        samplerate = SAMPLE_RATE,
        window_duration = WINDOW_DURATION,
        chunk_duration = CHUNK_DURATION,
        n_fft = N_FFT,
        n_mels = N_MELS,
        hop = HOP,
    ):
        self.samplerate = samplerate
        self.chunk_size = int(samplerate * chunk_duration)
        self.n_fft = n_fft
        self.window_size = int(samplerate * window_duration)
        self.n_mels = n_mels
        self.hop = hop

        self.sub_hop = n_fft // 4
        self.window = np.zeros(self.window_size)

        self.pitch = 0.0
        self.amp = 0.0
        self.note = None
        self.octave = None

        # Load the noise file for spectral substraction
        noise_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), NOISE_FILE)
        if os.path.exists(noise_path):
            self.car_noise = np.load(noise_path).astype(np.float32)
            expected = (n_fft // 2 + 1,)
            if self.car_noise.shape != expected:
                print(f"Warning: noise profile shape {self.car_noise.shape} != "
                      f"expected {expected}. Spectral subtraction disabled.")
                self.car_noise = None
            else:
                print(f"Loaded noise profile: shape={self.car_noise.shape}, "
                      f"mean={self.car_noise.mean():.0f}, max={self.car_noise.max():.0f}")
        else:
            print(f"Warning: {noise_path} not found. Spectral subtraction disabled.")
            self.car_noise = None

        # pre-computed hanning window used for log-Mel spectrogram
        self._sub_window = np.hanning(n_fft).astype(np.float32)

        # Musical notes for pitch to note conversion
        self.notes     = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        self.len_notes = len(self.notes)

        # Pre-computed mel filterbank and FFT frequencies for log-Mel spectrogram calculation
        self.mel_transform = FastMelSpec(sr=samplerate, n_fft=n_fft, hop=hop)

        self.recording = np.array([], dtype=np.int16)
        self.recording_cleaned = np.array([], dtype=np.int16)

# This applies spectral subtraction to the input audio using the preloaded noise profile, and returns the cleaned audio. 
# If no noise profile is available, it returns the original audio.
    def spectral_subtraction(self, audio, alpha: float = 1.0):

        if self.car_noise is None:
            return audio
        x= np.asarray(audio, dtype=np.float32).flatten()
        n_fft = self.n_fft
        hop = self.sub_hop
        window = self._sub_window

        n_frames = 1 + (len(x) - n_fft) // hop
        if n_frames < 1:
            return audio

        output = np.zeros(len(x), dtype=np.float32)
        norm = np.zeros(len(x), dtype=np.float32)

        for i in range(n_frames):
            start = i * hop
            frame = x[start:start + n_fft] * window

            spec = np.fft.rfft(frame)
            mag = np.abs(spec)
            phase = np.angle(spec)

            # Spectral subtraction
            cleaned_mag = np.maximum(mag - alpha * self.car_noise, 0.0)
            cleaned = np.fft.irfft(cleaned_mag * np.exp(1j * phase), n=n_fft)

            output[start:start + n_fft] += cleaned * window
            norm[start:start + n_fft]   += window ** 2

        floor = max(norm.max() * 0.01, 1e-6)
        norm = np.maximum(norm, floor)
        output = np.clip(output / norm, -32768.0, 32767.0)
        return output.astype(np.int16)

# This updates the internal window buffer with the new audio frames, and also append the frames to the recording buffer for later analysis.
    def update_window(self, frames):
        frames = frames.flatten()
        n = len(frames)
        self.window = np.roll(self.window, -n)
        self.window[-n:] = frames
        self.recording = np.concatenate([self.recording, frames])

# This is a helper function that converts a frequency in Hz to the corresponding musical note and octave.
    def freq_to_note(self, freq):
        if not np.isfinite(freq) or freq <= 0:
            return self.notes[0], 0
        note_number = round(12 * np.log2(freq / 440) + 49)
        note   = self.notes[(note_number - 1) % self.len_notes]

        # Calculate the octave number based on the note number.
        octave = (note_number + 8) // self.len_notes
        return note, octave

# This is the main method that takes in raw audio data, applies spectral subtraction, computes the mel-spectrogram, detects the pitch and note, and returns the mel-spectrogram as a numpy array of type float32.
    def CalcMel(self, soundData: np.ndarray) -> np.ndarray:
        sd = soundData.flatten()
        mel_spectrogram, freq, amplitude = self.mel_transform(audio=sd)
        self.note, self.octave = self.freq_to_note(freq)
        self.pitch, self.amp   = freq, amplitude
        return mel_spectrogram.astype(np.float32)
