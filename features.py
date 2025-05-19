import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import zoom

def create_spectrogram_data(input_wav_path, size=(300, 300)):
    """
    Generates a normalized spectrogram data matrix of fixed dimensions without any graphics.

    Args:
        input_wav_path (str): Path to the input WAV file.
        size (tuple): Desired (height, width) of the output matrix.

    Returns:
        np.ndarray: 2D array of shape `size` containing spectrogram values normalized to [0, 1].
    """
    # Read audio file
    sample_rate, samples = wavfile.read(input_wav_path)
    # If stereo, take the first channel
    if samples.ndim > 1:
        samples = samples[:, 0]

    # Compute spectrogram
    frequencies, times, Sxx = signal.spectrogram(samples, fs=sample_rate)
    # Convert to dB scale and normalize
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min())

    # Resize to desired dimensions
    original_shape = Sxx_norm.shape
    zoom_factors = (size[0] / original_shape[0], size[1] / original_shape[1])
    Sxx_resized = zoom(Sxx_norm, zoom_factors, order=1)

    return Sxx_resized