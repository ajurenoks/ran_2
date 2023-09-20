import csv
import os
import audioread
from tqdm import tqdm
import pandas as pd
import torch

rewrite_metadata = True
user_finetuning = False
specific_user = 'p304'
user_nr = specific_user[1:]
if torch.cuda.is_available():
    path = './dataset/' # GPU
else:
    path = '../../datasets/VCTK-Corpus-0.92/' #LOCAL

if user_finetuning:
    path_to_audio = f'./FreeVC/FreeVC-main/output/freevc_{specific_user}/'
    # faili šajā mapē ir pxxx_jjj_mic1_to_pyyy.wav , kur x - oriģinālais users, j- faila nr, y - specific user


if not os.path.exists(path):
    print("Wrong path!")
    exit()
# VCTK -> test/train/val
# test/train/val -> pxxx
# pxxx -> audio faili

if not user_finetuning:
    train_test_val_folders = os.listdir(path)
    for i in train_test_val_folders: # i = test/train/val
        if i == 'test' or i == 'train' or i =='val':
            speaker_folders = os.listdir(f'{path}{i}')
            print(f"{i} start:")
        else:
            print(f"{i} Not train/test/val")
            continue
        for j in tqdm(speaker_folders):  # j = pxxx
            path_to_txt = f'{path}txt/{j}' #LOCAL
            # path_to_txt = f'./FreeVC/data/txt/{j}' #GPU
            path_to_audio = f'{path}{i}/{j}'
            if not os.path.exists(path_to_txt):
                print("Wrong txt path!")
                print(path_to_txt)
                exit()
            if not os.path.exists(path_to_audio):
                print("Wrong audio path!")
                exit()
            txt_files = os.listdir(path_to_txt)
            audio_files = os.listdir(path_to_audio)
            if os.path.exists(f'{path_to_audio}/metadata.csv') and not rewrite_metadata:
                print(f"Metadata file for {j} already exists")
                if os.stat(f'{path_to_audio}/metadata.csv').st_size > 0:
                    if len(audio_files)-1 > len(pd.read_csv(f'{path_to_audio}/metadata.csv')):
                        print(" But there are some audio files missing")
                    else:
                        print(" And all files are there (by count)")
                        continue
                else:
                    print(" But the file is empty")

            f = open(f'{path_to_audio}/metadata.csv', 'w', newline='')

            writer = csv.writer(f)
            header = ['file_name', 'transcription', 'file_len', 'speaker_file']
            writer.writerow(header)


            for k in txt_files:
                file_name = f"{k.split('.')[0]}_mic1.flac"
                if not os.path.exists(f"{path_to_txt}/{k}"):
                    print(f"Weird, the path to txt does not exist for {path_to_txt}/{k}")
                transcript_file = open(f"{path_to_txt}/{k}", "r")
                transcription = transcript_file.readlines()[0].rstrip('\n').strip()
                if not os.path.exists(f"{path_to_audio}/{file_name}"):
                    print(f"Weird, the audio {file_name} doesn't exist on path: {path_to_audio}/{file_name}")
                    transcript_file.close()
                    continue
                audio_file = audioread.audio_open(f"{path_to_audio}/{file_name}")
                file_len = audio_file.duration
                speaker_file = k.split('.')[0]
                data = [file_name, transcription, file_len, speaker_file]
                writer.writerow(data)
                transcript_file.close()
                audio_file.close()
            f.close()
else:
    if not os.path.exists(path_to_audio):
        print(f"AUDIO PATH {path_to_audio} DOES NOT EXIST!!")
        exit()
    if os.path.exists(f"{path_to_audio}/metadata.csv"):
        os.remove(f"{path_to_audio}/metadata.csv")
    audio_files = os.listdir(path_to_audio)
    print(f"{path_to_audio}metadata.csv")
    if os.path.exists(f'{path_to_audio}/metadata.csv') and not rewrite_metadata:
        print(f"Metadata file for finetuned {specific_user} already exists")
        if os.stat(f'{path_to_audio}/metadata.csv').st_size > 0:
            if len(audio_files) - 1 > len(pd.read_csv(f'{path_to_audio}/metadata.csv')):
                print(" But there are some audio files missing")
            else:
                print(" And all files are there (by count)")
                # continue
        else:
            print(" But the file is empty")
    f = open(f'{path_to_audio}/metadata.csv', 'w', newline='')
    writer = csv.writer(f)
    header = ['file_name', 'transcription', 'file_len', 'speaker_file']
    writer.writerow(header)
    for i in tqdm(audio_files):
        try:
            if 'metadata' in i:
                continue
            speaker = i.split('_')[0]
            if speaker == 'p315':
                print("speaker p315 encountered O.o")
                continue
            if len(i.split("_")) <= 1:
                print(i)
            speaker_file = f"{speaker}_{i.split('_')[1]}"
            path_to_txt = f'./dataset/txt/{speaker}'  # GPU
            # txt_files = os.listdir(path_to_txt)
            txt_file = f"{path_to_txt}/{speaker_file}.txt"
            file_name = i
            if not os.path.exists(txt_file):
                print(f"Weird, the path to txt does not exist for {txt_file}")
                continue

            transcript_file = open(txt_file, "r")
            transcription = transcript_file.readlines()[0].rstrip('\n').strip()
            if not os.path.exists(f"{path_to_audio}/{file_name}"):
                print(f"Weird, the audio {file_name} doesn't exist on path: {path_to_audio}/{file_name}")
                transcript_file.close()
                continue
            audio_file = audioread.audio_open(f"{path_to_audio}/{file_name}")
            file_len = audio_file.duration
            data = [file_name, transcription, file_len, speaker_file]
            writer.writerow(data)
            transcript_file.close()
            audio_file.close()
        except:
            print("ERROR:", i)
            # audioread.audio_open("./FreeVC/FreeVC-main/output/freevc_p304/p364_099_mic1_to_p304.wav")
    f.close()






