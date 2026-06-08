# Acoustic Robot

A sound-controlled robot car operated by acoustic commands. A laptop captures
audio from a USB microphone, classifies it with a CNN, and sends motor
commands to a Raspberry Pi (PiCar-4WD) over a TCP socket. The Pi acts only as
a motor executor.

## Commands

| Sound       | Action                                                      |
|-------------|-------------------------------------------------------------|
| Clap        | Move forward                                                |
| Whistle     | Toggle drive state: forward → stop → backward → stop → ...  |
| Harmonica   | Turn left if pitch < 600 Hz, turn right if pitch ≥ 600 Hz   |

AcoucticLaptop/
├── requirements.txt          # All the dependencies
├── AudioStream.py            # Main entry point (run this)
├── AudioProcessor.py         # Rolling audio buffer, spectral subtraction, mel-spec
├── PitchExtraction.py        # Fast mel-spectrogram + HPS pitch detector
├── SoundClassifier.py        # AudioCNN architecture definition
├── RemoteMotorControl.py     # TCP client wrapping the motor interface
├── pi_server.py              # TCP server (runs on the Pi)
├── best_model.pt             # Trained CNN weights
├── mel_fb.npy                # Pre-computed mel filterbank
├── average_motor_noise.npy   # Motor noise profile for spectral subtraction
├── Training/                 # Training pipeline
│   ├── Data/                 # Per-class audio samples
│   │   ├── clap/
│   │   ├── harmonica/
│   │   ├── silence/
│   │   └── whistle/
│   ├── AudioAugmentationPipelin.py
│   ├── config.py
│   └── TrainClassifier.py
└── .venv/                    # Python virtual environment

## Setup

### Laptop

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio sounddevice numpy scipy librosa
```

On macOS, `sounddevice` requires PortAudio:

```bash
brew install portaudio
```

### Raspberry Pi

Copy `pi_server.py` to the Pi:

```bash
scp pi_server.py pi@10.42.0.1:~/
```

The Pi must have the `picar_4wd` motor library installed (default on PiCar-4WD
images).

## Running

### 1. Start the Pi server

SSH into the Pi:

```bash
ssh pi@10.42.0.1
python3 pi_server.py
```

For dry-run testing without hardware, append `--sim`.

### 2. Identify the microphone on the laptop

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Note the index of the ReSpeaker (or whichever USB mic is in use).

### 3. Start the laptop pipeline

```bash
python AudioStream.py --pi 10.42.0.1 --d <mic_index> --alpha 3.0 --model best_model.pt
```

Press **Ctrl+C** to stop. On exit, raw and noise-subtracted recordings are
saved as `background_recorded.wav` and `cleaned_recorded.wav`.

### Command-line flags

| Flag             | Default              | Description                            |
|------------------|----------------------|----------------------------------------|
| `--pi`           | `10.42.0.1`          | Pi IP address                          |
| `--pi_port`      | `5005`               | TCP port                               |
| `--d`            | —                    | Microphone device index                |
| `--alpha`        | `2.0`                | Spectral subtraction strength          |
| `--model`        | `finetuned_model.pt` | Path to model weights                  |
| `--no_motors`    | off                  | Run without sending commands to the Pi |
| `--no_save`      | off                  | Disable saving .wav recordings on exit |
| `--callback_time`| `0.25`               | Audio chunk duration in seconds        |

## Architecture

The laptop is responsible for all audio processing and inference; the Pi only
executes motor commands.

```
LAPTOP                                          PI
──────                                          ──
ReSpeaker USB mic                               pi_server.py
   ↓                                                ↓
AudioStream.py                                 picar_4wd library
   ↓ spectral subtraction                          ↑
   ↓ mel-spectrogram                               │
   ↓ AudioCNN inference                            │
   ↓ state machine + gating                        │
   ↓ pitch-based steering decision                 │
   └──── TCP socket (port 5005) ──────────────────┘
        sends "forward" / "backward" / "left" / "right" / "stop"
```

### State machine

- **Drive state** (`stopped` / `forward` / `backward`) — changed only by clap
  and whistle. Three silent frames are required between consecutive drive
  commands.
- **Steering overlay** — harmonica fires immediately, no gating. A turn ends
  after three consecutive non-harmonica frames; the prior drive state then
  resumes.
- During an active turn, clap and whistle are ignored. After a harmonica, the
  three following silent frames also block clap and whistle to suppress
  residual misclassifications in the harmonica's tail.

### Pitch detection

Pitch is estimated via Harmonic Product Spectrum on the full one-second
window. HPS reinforces the true fundamental by multiplying the magnitude
spectrum with downsampled copies of itself, mitigating the octave-jumping
that plain FFT-argmax produces on harmonic-rich instruments such as the
tremolo harmonica.

The detected frequency is split into LEFT or RIGHT at `PITCH_HIGH_HZ_THRESHOLD`
(default 600 Hz), matching the layout of a C tremolo harmonica.

## Training

To retrain the classifier on the data in `Training/Data/`:

```bash
cd Training
python TrainClassifier.py
give your input for the retraining model as your_model_name.pt
```

The best model (lowest validation loss) is saved to `../<your_model_name>.pt`,
overwriting the current weights used by the live system.

When retrained, please check the order of the class names that it prints in the stdout, they must match with the class names that are in AudioStream.py, otherwise the actions will not be linked to the right class labels, and it won't perform as intended.

Change the variable "CLASSES" in line no.34 in AudioStream.py to the correct order from your retrained model's order. For example, the standard class order in AudioStream.py when best_model.pt is used, its {0: "silence", 1: "harmonica", 2: "whistle", 3: "clap"}, and lets say, your retrained the output of the class order at the beginning of the training was [harmonica, clap, silence, whistle], then change line no.34 to {0: "harmonica", 1: "clap", 2: "silence", 3: "whistle"}.


If you want to test your new model run the following command:
```bash
python AudioStream.py --pi 10.42.0.1 --d <mic_index> --alpha 3.0 --model <your_model_name.pt>
```

## Network setup

The default address `10.42.0.1` is the gateway assigned by Linux
NetworkManager when sharing a connection over Ethernet or USB tethering.
Verify reachability with:

```bash
ping 10.42.0.1
nc 10.42.0.1 5005
```

In `nc`, typing `ping` should return `pong`.