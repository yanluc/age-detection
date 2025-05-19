import soundfile as sf
import os
import pandas as pd
import librosa 
from pydub import AudioSegment

ROOT_PATH = "./cv-corpus-21.0-2025-03-14/pl/clips"
fileListInput = pd.read_csv('prepared.tsv', sep='\t')
fileListOutput = pd.DataFrame(columns=['path', 'age', 'gender_female', 'gender_male', 'if_added'])

# Function to split and save audio segments with overlap
def split_and_save_audio_segments(audio_path, seg_length, overlap, output_folder, file_idx, file):
    """
    Splits an audio file into overlapping segments and saves them as .ogg files.
    
    Args:
        audio_path (str): Path to the input audio file.
        seg_length (int): Length of each segment in seconds.
        overlap (float): Fraction of segment length for overlap (0.0 to 1.0).
        output_folder (str): Folder to save the segmented audio files.
        file_idx (int): Unique index for naming the output files.
    """
    
    # Load the audio file
    try:
        y, sr = librosa.load(audio_path, sr=None)  # Faster than librosa.load()
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return []
        

    # Calculate segment & overlap sizes in samples
    segment_samples = seg_length * sr
    overlap_samples = int(segment_samples * overlap)

    num_samples = len(y)
    start = 0
    segment_counter = 0
    saved_files = []

    # Extract and save audio fragments
    while start + segment_samples <= num_samples:
        end = start + segment_samples
        segment = y[start:end]

         # Convert numpy array to AudioSegment
        audio_segment = AudioSegment(
            segment.tobytes(),
            frame_rate=sr,
            sample_width=segment.dtype.itemsize,
            channels=1 if segment.ndim == 1 else segment.shape[1]
        )

        # Construct the filename
        sample_name = f"sample_{file_idx}_{segment_counter + 1}.mp3"
        filename = os.path.join(output_folder, sample_name)

        # Save as MP3
        audio_segment.export(filename, format="mp3")
        print(f"Zapisano: {filename}")

        saved_files.append(filename)
        fileListOutput.loc[len(fileListOutput)] = [
            filename,
            file['age'],
            file['gender_female'],
            file['gender_male'],
            file['if_added']
        ]

        segment_counter += 1
        start += overlap_samples  # Move by the overlap size

    return saved_files  # Return list of saved filenames




def process_audio_files(in_folder, out_folder, seg_len, overlap):
    for _, file in fileListInput.iterrows():
        audio_path= in_folder+'/'+file['path']
        saved_files = split_and_save_audio_segments(audio_path, seg_len, overlap, out_folder, file['path'], file)



# Usage example
out_folder = "segmented_soundscapes"
seg_len = 1  # Segment length in seconds
overlap = 0.5  # 50% overlap

os.makedirs(out_folder, exist_ok=True)
process_audio_files(ROOT_PATH, out_folder, seg_len, overlap)

fileListOutput.to_csv('output_segments.csv', index=False)


