
import os
import pandas as pd
import librosa 
import numpy as np
from pydub import AudioSegment
import csv
import argparse


ROOT_PATH = "./cv-corpus-21.0-2025-03-14/pl/clips"

acumulator = np.array([], dtype=np.float32)

def save_segment(segment, sr, output_folder, age, gf, gm, if_aded):

    '''
    audio_segment = AudioSegment(
            segment.tobytes(),
            frame_rate=sr,
            sample_width=segment.dtype.itemsize,
            channels=1 if segment.ndim == 1 else segment.shape[1]
        )
    '''

    segment_int16 = np.int16(segment / np.max(np.abs(segment)) * 32767)
    audio_segment = AudioSegment(
        segment_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 2 bajty dla int16
        channels=1
    )
    
    sample_name = f"sample_{hex(hash(segment.tobytes()))}_{age}_{gf}_{gm}.mp3"
    filename = os.path.join(output_folder, sample_name)

     # Save as MP3
    audio_segment.export(filename, format="mp3")
    print(f"Zapisano: {filename}")

    csv_file = "output_segments.csv"

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)

        # add columns name if file not exist
        if not file_exists:
            writer.writerow(['filename', 'age', 'gender_female', 'gender_male', 'if_added'])

        # save data
        writer.writerow([
            sample_name,
            age,
            gf,
            gm,
            if_aded
        ])

# Function to split and save audio segments with overlap
def split_and_save_audio_segments(audio_path, seg_length, output_folder, file, lastFile):

    global acumulator

    if file['age'] != lastFile['age'] or file['gender_female'] != lastFile['gender_female'] or file['gender_male'] != lastFile['gender_male']:
        acumulator = np.array([], dtype=np.float32)

    
    # Load the audio file
    try:
        y, sr = librosa.load(audio_path, sr=44100)  # Faster than librosa.load()
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return []
        

    # Calculate segment & overlap sizes in samples
    segment_samples = seg_length * sr
  
    num_samples = len(y)
    start = 0
    segment_counter = 0


    # Extract and save audio fragments
    while start + (segment_samples-len(acumulator)) <= num_samples:
        end = start + (segment_samples-len(acumulator))
        segment = np.concatenate((y[start:end], acumulator))

        save_segment(segment, sr, output_folder, file['age'], file['gender_female'], file['gender_male'], file['if_added'])

        segment_counter += 1
        #start += overlap_samples  # Move by the overlap size
        start += (segment_samples-len(acumulator))
        acumulator = np.array([], dtype=np.float32)


    if start + (segment_samples-len(acumulator)) > num_samples:
        segment = y[start:]
        acumulator = np.concatenate((acumulator, segment))


def process_audio_files(in_folder, out_folder, seg_len, fileListInput):
   fileListInput = fileListInput.sort_values(by=['age', 'gender_female', 'gender_male']) 

   for i in range(0, len(fileListInput)):
       file = fileListInput.iloc[i]
       lastFile = None
       if i>0:
           lastFile=fileListInput.iloc[i-1]
       else:
           lastFile = fileListInput.iloc[i]
           
       audio_path= in_folder+'/'+file['path']
       split_and_save_audio_segments(audio_path, seg_len, out_folder, file, lastFile)
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentuj pliki audio bez nakładania fragmentów.")
    parser.add_argument('--in_folder', type=str, default="./cv-corpus-21.0-2025-03-14/pl/clips", help="Ścieżka do folderu z plikami audio.")
    parser.add_argument('--out_folder', type=str, default="segmented_soundscapes", help="Ścieżka do folderu wyjściowego.")
    parser.add_argument('--seg_length', type=int, default=20, help="Długość segmentu w sekundach (domyślnie: 1).")

    args = parser.parse_args()
    ROOT_PATH=args.in_folder

    fileListInput = pd.read_csv('prepared.tsv', sep='\t')

    os.makedirs(args.out_folder, exist_ok=True)
    process_audio_files(ROOT_PATH, args.out_folder, args.seg_length, fileListInput)




