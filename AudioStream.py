import argparse
import os
import queue
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import torch
from scipy.io.wavfile import write

from AudioProcessor import AudioProcessor
from RemoteMotorControl import RemoteMotorControl
from SoundClassifier import AudioCNN

CONFIDENCE_THRESHOLD = 0.90 # Minimum CNN softmax probability to accept a prediction
STABILITY_IDLE = 1 # Consecutive matching frames needed to act when motors are off
STABILITY_MOTORS_ON = 2 # Same as STABILITY_IDLE but stricter when motors are running 
SILENCE_FRAMES_BETWEEN_ACTIONS = 2 # Silent frames required between two clap/whistle commands
POST_HARMONICA_SILENCE = 3 # Silent frames required after a harmonica before clap/whistle re-enable
TURN_TIMEOUT_FRAMES = 3 # Non-harmonica frames after which an active turn ends and drive state resumes
INITIAL_TIMEOUT = 8 # Startup frames to discard while the audio buffer fills
DEFAULT_ALPHA = 3.0 # Strength of spectral subtraction 
DEFAULT_SAMPLERATE = 15_872 # Input audio sample rate in Hz
DEFAULT_CALLBACK = 0.25 # Audio chunk size in seconds 
DEFAULT_PI_HOST = "10.42.0.1" # Pi's IP address
DEFAULT_PI_PORT = 5005 # TCP port for the motor command server
MODEL_FILENAME = "New_Model.pt" # Path to the trained .pt weights file
N_CLASSES = 4 # Number of output classes the CNN predicts
VOLUME_THRESHOLD = 500.0 # RMS below this is treated as silence
PITCH_HIGH_HZ_THRESHOLD = 600.0 # Harmonica notes above this Hz will turn the robot to right, below it turns left
CLASSES = {0: "silence", 1: "harmonica", 2: "whistle", 3: "clap"} # Index - to - name mapping for CNN output classes
SILENCE_LABEL_IDX = 2 # The class index that means "silence" (= 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int,   help="Microphone device index")
    p.add_argument("--samplerate", type=int,   default=DEFAULT_SAMPLERATE)
    p.add_argument("--channels", type=int,   default=1)
    p.add_argument("--blocksize", type=int,   default=512)
    p.add_argument("--callback_time",type=float, default=DEFAULT_CALLBACK)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--pi", type=str,   default=DEFAULT_PI_HOST)
    p.add_argument("--pi_port", type=int,   default=DEFAULT_PI_PORT)
    p.add_argument("--no_motors", action="store_true")
    p.add_argument("--model", type=str,   default=MODEL_FILENAME)
    return p.parse_args()

def configure_sounddevice(args: argparse.Namespace) -> None:
    sd.default.samplerate = args.samplerate
    sd.default.channels = args.channels
    sd.default.blocksize = int(args.samplerate * args.callback_time)
    sd.default.dtype = np.int16

# This is a helper function that builds the AudioProcessor object with the appropriate parameters based on the command-line arguments.
def build_audio_processor(args: argparse.Namespace) -> AudioProcessor:
    return AudioProcessor(
        samplerate = args.samplerate,
        chunk_duration = args.callback_time,
        n_fft = 512,
        n_mels = 62,
        hop = 512,
    )

# This is a helper function that loads the pretrained PyTorch model.
def load_model(args: argparse.Namespace) -> AudioCNN:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    device = torch.device("cpu")
    model = AudioCNN(n_classes=N_CLASSES)

    # Load the model state dict from the .pt file, ensuring it's compatible with CPU even if it was trained on GPU and set the model to evaluation mode.
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f" PyTorch model loaded from {os.path.basename(model_path)} (device=cpu)")
    return model

# This is a helper function that builds the RemoteMotorControl object with the appropriate parameters based on the command-line arguments.
def build_motor_control(args: argparse.Namespace) -> RemoteMotorControl:
    return RemoteMotorControl(
        host= args.pi,
        port= args.pi_port,
        power_forward= 20,
        power_backward = 20,
        power_steer= 5,
        high_notes= {'D#', 'E', 'F', 'F#', 'G', 'G#'},
        motor_controls = not args.no_motors,
    )

# This is a helper function that reports whether the noise profile was successfully loaded and spectral subtraction is active, based on the presence of the noise profile in the AudioProcessor object.
def report_noise_profile(ap: AudioProcessor, alpha: float) -> None:
    if ap.car_noise is None:
        print("\n average_motor_noise.npy not found — spectral subtraction DISABLED.\n")
    else:
        print(f" Noise profile loaded — spectral subtraction active (alpha={alpha})")

@dataclass
class CarState:
    last_pred: Optional[int] = None
    stable_count: int = 0
    initial_timeout: int = INITIAL_TIMEOUT
    drive_state: str = "stopped"
    is_turning:bool = False
    non_harmonica_streak:int= 0
    frames_since_last_drive_cmd: int   = 10**6
    frames_since_harmonica:int   = 10**6

# This is a helper function that sends a motor command to the Pi server by calling the _send method of the RemoteMotorControl object. It also checks if motor controls are enabled before sending the command.
def mix_to_mono(chunk: np.ndarray) -> np.ndarray:
    if chunk.ndim == 2 and chunk.shape[1] > 1:
        return chunk.mean(axis=1, keepdims=True).astype(np.int16)
    return chunk

# Returns the audio with motor noise subtracted out, but only when the motors are actually running.
def apply_spectral_subtraction(
    ap: AudioProcessor, motors_active: bool, alpha: float,
) -> np.ndarray:
    if motors_active:
        return ap.spectral_subtraction(ap.window, alpha=alpha)
    return ap.window

# Computes the root mean square of the audio, a measure of overall loudness.
def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

# Classifies one audio frame into clap/harmonica/silence/whistle. Returns (pred_idx, confidence, label).
def classify_frame(
    audio: np.ndarray, ap: AudioProcessor, model: AudioCNN, rms: float,
) -> Tuple[int, float, str]:
    if rms <= VOLUME_THRESHOLD:
        ap.note = None
        ap.octave = None
        return SILENCE_LABEL_IDX, 1.0, "silence"
    # Normalize the audio to [-1, 1], compute the log-Mel spectrogram, and run it through the CNN to get class probabilities.
    #audio_norm = audio.astype(np.float32)
    #audio_norm = audio_norm / (float(np.max(np.abs(audio_norm))) + 1e-9)
    mel_data = ap.CalcMel(audio)
    with torch.no_grad():
        x = torch.from_numpy(mel_data).float().unsqueeze(0).unsqueeze(0)
        probs = model(x).squeeze(0).numpy()
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    label = CLASSES[pred]

    if label in ("whistle", "harmonica") and ap.octave is not None:
        if ap.octave < 2 or ap.octave > 7:
            return SILENCE_LABEL_IDX, 1.0, "silence"

    return pred, confidence, label

# This is a helper function that updates the stability count of the car based on the predicted class label.
def update_stability(state: CarState, pred: int) -> int:
    if pred == state.last_pred:
        state.stable_count += 1
    else:
        state.stable_count = 1
        state.last_pred    = pred
    return state.stable_count

# Sends a drive command (forward / backward / stop) directly to the motor controller.
def issue_drive(car: RemoteMotorControl, action: str) -> None:
    car._execute_action(action)

# Sends a turn command (left / right) directly to the motor controller.
def issue_turn(car: RemoteMotorControl, direction: str) -> None:
    car._execute_action(direction)

# This is the main method that gets called with the predicted class label and detected note.
def resume_drive_state(car: RemoteMotorControl, drive_state: str) -> None:
    if drive_state == "forward":
        issue_drive(car, "forward")
    elif drive_state == "backward":
        issue_drive(car, "backward")
    else:
        issue_drive(car, "stop")

# This is a helper function that decides the next drive state (forward/backward/stopped) based on the current drive state and the predicted class label (clap/whistle). 
# It returns the new drive state or None if no change is needed.
def decide_drive_transition(current: str, label: str) -> Optional[str]:
    if label == "clap":
        return "forward" if current != "forward" else None
    if label == "whistle":
        if current == "forward":  return "stopped"
        if current == "backward": return "stopped"
        return "backward"
    return None

# This is a helper function that checks if the detected pitch is considered "high" based on a predefined threshold.
def is_high_pitch(pitch_hz: float, threshold_hz: float = PITCH_HIGH_HZ_THRESHOLD) -> bool:
    return pitch_hz >= threshold_hz

# This is the main method that gets called with the predicted class label and detected note.
def try_handle_frame(
    state: CarState,
    car: RemoteMotorControl,
    label: str,
    confidence: float,
    note: Optional[str],
    pitch_hz: float,
    motors_active: bool,
) -> Tuple[Optional[str], Optional[str]]:
    is_silent = (label == "silence")

    if is_silent:
        state.frames_since_last_drive_cmd += 1
        state.frames_since_harmonica += 1
    if state.is_turning:
        if label == "harmonica" and confidence >= CONFIDENCE_THRESHOLD:
            state.non_harmonica_streak = 0
        else:
            state.non_harmonica_streak += 1
            if state.non_harmonica_streak >= TURN_TIMEOUT_FRAMES:
                state.is_turning = False
                resume_drive_state(car, state.drive_state)
                return f"end-turn → {state.drive_state}", None
    if not is_silent:
        if confidence < CONFIDENCE_THRESHOLD:
            return None, None
        required = STABILITY_MOTORS_ON if motors_active else STABILITY_IDLE
        if state.stable_count < required:
            return None, None
    if label == "harmonica":
        direction = "right" if is_high_pitch(pitch_hz) else "left"
        issue_turn(car, direction)
        state.is_turning = True
        state.non_harmonica_streak = 0
        state.frames_since_harmonica = 0
        return f"turn {direction} ({pitch_hz:.0f}Hz)", None
    if state.is_turning and label in ("clap", "whistle"):
        return None, "ignored (turn active)"
    if (label in ("clap", "whistle")
            and state.frames_since_harmonica < POST_HARMONICA_SILENCE):
        need = POST_HARMONICA_SILENCE - state.frames_since_harmonica
        return None, f"post-harmonica window (need {need} more silent)"
    if (label in ("clap", "whistle")
            and state.frames_since_last_drive_cmd < SILENCE_FRAMES_BETWEEN_ACTIONS):
        need = SILENCE_FRAMES_BETWEEN_ACTIONS - state.frames_since_last_drive_cmd
        return None, f"need {need} more silent frames"
    if label in ("clap", "whistle"):
        new_state = decide_drive_transition(state.drive_state, label)
        if new_state is None:
            return None, "already in target state"
        action = "stop" if new_state == "stopped" else new_state
        issue_drive(car, action)
        state.drive_state = new_state
        state.frames_since_last_drive_cmd = 0
        return action, None

    return None, None

# Prints a one-line summary of an actionable frame to the console. 
# Skips silence frames (unless they triggered an action like ending a turn).
def log_frame(
    label: str, confidence: float, rms: float, state: CarState,
    required_stability: int, elapsed: float, motors_active: bool,
    action: Optional[str], blocked_reason: Optional[str],
    note: Optional[str], octave: Optional[int],
) -> None:
    if label == "silence" and action is None:
        return  # don't spam silence
    if label != "silence" and confidence < CONFIDENCE_THRESHOLD:
        return
    ns = "noise-sub" if motors_active else "raw      "
    gate_tag = f" [BLOCKED: {blocked_reason}]" if blocked_reason else ""
    pitch_tag = f"pitch: {octave}{note}" if note else "pitch: ---"
    print(
        f"[{ns}] {label:<10} | "
        f"{pitch_tag:<12} | "
        f"conf: {confidence:.2f} | "
        f"rms: {rms:6.0f} | "
        f"stab: {state.stable_count}/{required_stability} | "
        f"drive: {state.drive_state:<8} | "
        f"turn: {'Y' if state.is_turning else 'N'} | "
        f"t: {elapsed*1000:5.1f}ms | "
        f"action: {action}{gate_tag}"
    )

# Prints all the active hyperparameters and runtime config in one banner when the program starts.
def print_startup_banner(args: argparse.Namespace) -> None:
    print(
        f"\nListening...  alpha={args.alpha} | "
        f"confidence≥{CONFIDENCE_THRESHOLD} | "
        f"stability idle={STABILITY_IDLE}/motors={STABILITY_MOTORS_ON} | "
        f"silence-gate={SILENCE_FRAMES_BETWEEN_ACTIONS} | "
        f"post-harm={POST_HARMONICA_SILENCE} | "
        f"turn-timeout={TURN_TIMEOUT_FRAMES} | "
        f"pitch-split={PITCH_HIGH_HZ_THRESHOLD:.0f}Hz | "
        f"pi={args.pi}:{args.pi_port} | "
        f"motors={'OFF' if args.no_motors else 'ON'}\n"
    )

# Cleans up on Ctrl+C: stops the motors, closes the connection to the Pi.
# Overwrites the two recording files (background_recorded.wav and cleaned_recorded.wav) with this session's audio.
def shutdown(
    car: RemoteMotorControl, ap: AudioProcessor, samplerate: int, save_recordings: bool,
) -> None:
    print("\nStopping …")
    car._execute_action("stop")
    car.close()
    if not save_recordings:
        return
    write(f"Recorded_Background.wav", samplerate, ap.recording)
    print(f"Saved Recorded_Background.wav")
    write(f"Recorded_Cleaned.wav", samplerate, ap.recording_cleaned)
    print(f"Saved Recorded_Cleaned.wav")

# Processes one audio chunk through the full pipeline — from raw mic input to motor command.
def process_one_frame(
    chunk: np.ndarray, ap: AudioProcessor, model: AudioCNN,
    car: RemoteMotorControl, state: CarState, alpha: float,
) -> None:
    chunk = mix_to_mono(chunk)
    t0 = perf_counter()
    ap.update_window(chunk)
    if state.initial_timeout > 0:
        state.initial_timeout -= 1
        return
    motors_active = car.forward or car.backward or car.turning
    audio = apply_spectral_subtraction(ap, motors_active, alpha)
    # audio=ap.window
    ap.recording_cleaned = np.concatenate(
        [ap.recording_cleaned, audio[-ap.chunk_size:].astype(np.int16)]
    )
    rms = compute_rms(audio)
    pred, confidence, label = classify_frame(audio, ap, model, rms)
    elapsed = perf_counter() - t0

    update_stability(state, pred)
    required_stability = STABILITY_MOTORS_ON if motors_active else STABILITY_IDLE

    pitch_hz = float(ap.pitch) if ap.pitch is not None else 0.0
    action, blocked_reason = try_handle_frame(
        state, car, label, confidence, ap.note, pitch_hz, motors_active,
    )

    log_frame(
        label, confidence, rms, state, required_stability,
        elapsed, motors_active, action, blocked_reason, ap.note, ap.octave,
    )

# Boots up the system and runs the main listening loop until we stop it.
def main() -> None:
    args = parse_args()
    configure_sounddevice(args)
    ap = build_audio_processor(args)
    model = load_model(args)
    car = build_motor_control(args)
    report_noise_profile(ap, args.alpha)
    state = CarState()
    audio_queue = queue.Queue()

    def sd_callback(indata, frames, cb_time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())
    print_startup_banner(args)
    with sd.InputStream(device=args.d, channels=args.channels, callback=sd_callback):
        try:
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                process_one_frame(chunk, ap, model, car, state, args.alpha)
        except KeyboardInterrupt:
            shutdown(car, ap, args.samplerate, save_recordings=not args.no_save)

if __name__ == "__main__":
    main()