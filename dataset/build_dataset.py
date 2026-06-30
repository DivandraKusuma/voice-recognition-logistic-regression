import os
import pandas as pd
import concurrent.futures
from tqdm import tqdm 
from prepocessing.voice_preprocessing_ml import preprocess_audio_ml
from features.feature_extractor_ml import extract_features_ml

DATASET_PATH = "dataset"
OUTPUT_CSV = "dataset_fitur_audio.csv"

def process_single_file(file_info):
    file_path = file_info['path']
    file_name = file_info['filename']
    label = file_info['label']
    
    #panggil fungsi preprocessing
    y, sr = preprocess_audio_ml(file_path)
    
    if y is not None:
        #panggil fungsi ekstraksi fitur
        features = extract_features_ml(y, sr)
        features['filename'] = file_name
        features['label'] = label
        
        return features
        
    return None

def main():
    print("membuat tabel")
    
    all_tasks = []
    for label in ["male", "female"]:
        folder_path = os.path.join(DATASET_PATH, label)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.wav') or f.endswith('.mp3')]
            for f in files:
                all_tasks.append({
                    'path': os.path.join(folder_path, f),
                    'filename': f,
                    'label': label
                })
                
        
    print(f"Total file: {len(all_tasks)}")
    
    all_data = []
    
    #multithread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_file, all_tasks), total=len(all_tasks)))
        
    #hasil
    for res in results:
        if res is not None:
            all_data.append(res)
            
    print("done")
    
    #create tabel
    if len(all_data) > 0:
        df = pd.DataFrame(all_data)
        
        #pindahkan kolom nama file dan label
        kolom_lain = [c for c in df.columns if c not in ['filename', 'label']]
        df = df[['filename', 'label'] + kolom_lain]
        
        #simpan
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Tabel :  {OUTPUT_CSV}")
    else:
        print("Gagal!")

if __name__ == "__main__":
    main()
