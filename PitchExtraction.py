from __future__ import annotations
import numpy as np

PITCH_MIN_HZ = 80.0 # The lowest frequency the pitch detector will consider
PITCH_MAX_HZ = 2000.0 # The highest frequency the pitch detector will consider
HPS_HARMONICS = 4 # Number of harmonics to multiply in the Harmonic Product Spectrum for pitch detection

class FastMelSpec:
    def __init__(
        self,
        mel_fb_path: str = "mel_fb.npy",
        sr: int = 15_872,
        n_fft: int = 512,
        hop: int = 512,
        top_db: float = 90.0,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.top_db = top_db
        self.window = np.hanning(n_fft).astype(np.float32)
        self.mel_fb = np.load(mel_fb_path).astype(np.float32)
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32)

# This computes a Short-Time Fourier Transform (STFT) and returns its power spectrogram, basically,
# What frequencies are present at each moment in time across the audio clip is given by this function.
# This is the foundation of the mel-spectrogram that we feed into the CNN.
    def _power_stft(self, x: np.ndarray) -> np.ndarray:

        x = np.pad(x, self.n_fft // 2, mode='reflect')
        n_frames = 1 + (len(x) - self.n_fft) // self.hop
        frames = np.lib.stride_tricks.as_strided(
            x,
            shape=(n_frames, self.n_fft),
            strides=(x.strides[0] * self.hop, x.strides[0]),
            writeable=False,
        )
        spec = np.fft.rfft(frames * self.window, axis=-1)
        power = spec.real ** 2 + spec.imag ** 2

        return power.T.astype(np.float32)

# Estimates the fundamental frequency of "x" using Harmonic Product Spectrum and then returns the peak bin's frequency and its magnitude.
    def _detect_pitch(self, x: np.ndarray) -> tuple[float, float]:
        n = len(x)
        if n < 64:
            return 0.0, 0.0
        n_fft_pitch = 1 << max(6, (n - 1).bit_length())
        pad = np.zeros(n_fft_pitch, dtype=np.float32)
        window = np.hanning(n).astype(np.float32)
        pad[:n] = x * window

        spec = np.fft.rfft(pad)
        mag  = np.abs(spec).astype(np.float32)
        freqs = np.fft.rfftfreq(n_fft_pitch, 1.0 / self.sr).astype(np.float32)
        hps_len = len(mag) // HPS_HARMONICS
        if hps_len < 4:
            return 0.0, 0.0

        # Calculate the Harmonic Product Spectrum (HPS) by multiplying the magnitude spectrum downsampled by integer factors, which helps to enhance the fundamental frequency peak.
        hps = mag[:hps_len].copy()
        for k in range(2, HPS_HARMONICS + 1):
            hps *= mag[::k][:hps_len]
        
        # Mask out frequencies outside the desired pitch range and find the peak in the HPS, which corresponds to the estimated fundamental frequency. Returns both the frequency and its amplitude.
        hps_freqs = freqs[:hps_len]
        mask = (hps_freqs >= PITCH_MIN_HZ) & (hps_freqs <= PITCH_MAX_HZ)
        hps_masked = np.where(mask, hps, 0.0)

        if not np.any(hps_masked > 0):
            return 0.0, 0.0
        peak_bin = int(np.argmax(hps_masked))
        pitch_hz = float(hps_freqs[peak_bin])
        amplitude = float(mag[peak_bin])

        return pitch_hz, amplitude

# This is the main method that gets called with the raw audio data. It normalizes the audio
# Computes the log-Mel spectrogram, detects the pitch, and returns all of this information for use in classification and motor control.
    def __call__(self, audio: np.ndarray):
        x = np.ascontiguousarray(audio, dtype=np.float32).ravel()
        x = x / (np.max(np.abs(x)) + 1e-9)
        power = self._power_stft(x)
        mel = self.mel_fb @ power
        log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - self.top_db)
        pitch_hz, amplitude = self._detect_pitch(x)

        return log_mel.astype(np.float32), pitch_hz, amplitude