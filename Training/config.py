from dataclasses import dataclass, field
from typing import List, Tuple 


@dataclass
class AugmentConfig:

    # Noise
    noise_prob:       float = 0.5
    noise_snr_range:  Tuple[float, float] = (10.0, 30.0)   # dB  (higher = cleaner)
    # Which noise types to sample from. 
    noise_types:      List[str] = field(default_factory=lambda: ['white', 'pink'])
    db_reduction:       int = 20

 
    # Frame shift
    time_shift_prob:  float = 0.3
    time_shift_range: Tuple[float, float] = (-0.1, 0.1)

    # Pitch shift
    pitch_shift_prob:  float = 0.4
    pitch_shift_range: Tuple[float, float] = (-3.0, 3.0)   # semitones
 
    # Volume scaling
    volume_scale_prob:  float = 0.5
    volume_gain_range:  Tuple[float, float] = (0.5, 1.5)   # linear gain
 
    # SpecAugment
    spec_augment_prob:  float = 0.5
    n_freq_masks:       int   = 1
    freq_mask_param:    int   = 1    # max mel bands to zero out per mask
    n_time_masks:       int   = 1
    time_mask_param:    int   = 5   # max time frames to zero out per mask

    # Mixup
    alpha:              float = 0.2
    mixup_prob:         float = 0.3
