import librosa
import skimage.io
import numpy as np
import cv2
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled
def create_spectrogram(audio_path, output_path, target_size=(256, 256)):
    # Load an audio file
    y, sr = librosa.load(audio_path, sr=None)  

    # Parameters for Mel spectrogram
    n_fft = 1024       
    hop_length = 512    
    n_mels = 128       
    fmin = 20          
    fmax = sr // 2     

    # Generate Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, 
                                       n_mels=n_mels, fmin=fmin, fmax=fmax)
    S = np.log(S + 1e-9)  # Convert to log scale (avoid log(0) errors)

    # Normalize and invert spectrogram
    S = scale_minmax(S, 0, 255).astype(np.uint8)
    S = np.flip(S, axis=0)  # Put low frequencies at the bottom
    S = 255 - S  # Invert: Black = More Energy

    # Convert to 3-channel grayscale image (for CNN compatibility)
    S = cv2.merge([S, S, S])

    S_resized = cv2.resize(S, [256, 256], interpolation=cv2.INTER_AREA)  # Resize

    # Save as PNG
    skimage.io.imsave(output_path, S_resized)