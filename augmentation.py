import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import random
import librosa
import soundfile as sf

def random_augment(samples: np.ndarray, sample_rate: int):
    min_gauss_amplitude = random.uniform(0.001, 0.01)
    max_gauss_amplitude = random.uniform(0.01, 0.05)
    
    augment = Compose([
        AddGaussianNoise(
            min_amplitude=min_gauss_amplitude, 
            max_amplitude=max_gauss_amplitude, 
            p=1),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(p=0.5)
    ])
    return augment(samples=samples, sample_rate=sample_rate)

if __name__ == "__main__":
    input_path = ""
    samples, sample_rate = librosa.load(input_path, sr=None)

    augmented_samples = random_augment(samples, sample_rate)

    output_path = ""
    sf.write(output_path, augmented_samples, sample_rate)




