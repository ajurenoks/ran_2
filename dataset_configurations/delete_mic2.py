import os
from tqdm import tqdm

path = '../dataset/' # GPU
# path = '../../datasets/VCTK-Corpus-0.92/' #LOCAL
if not os.path.exists(path):
    print("Wrong path!")
    exit()
# VCTK -> test/train/val
# test/train/val -> pxxx
# pxxx -> audio faili
train_test_val_folders = os.listdir(path)
for i in train_test_val_folders: # i = test/train/val
    if i == 'test' or i == 'train' or i =='val':
        speaker_folders = os.listdir(f'{path}{i}')
    else:
        print(f"{i} Not train/test/val")
        continue
    for j in tqdm(speaker_folders):  # j = pxxx
        audio_files = os.listdir(f'{path}{i}/{j}')
        # audio_files = os.listdir(path)
        for k in audio_files:  # k = audio faili pxxx mapītē
            if 'mic2' in k:
                os.remove(f'{path}{i}/{j}/{k}')


print("All done! =)")


