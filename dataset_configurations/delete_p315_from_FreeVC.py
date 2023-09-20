import os
from tqdm import tqdm
specific_user = 'p287'
user_nr = specific_user[1:]
path = f'./FreeVC/output/freevc_{user_nr}/'
if not os.path.exists(path):
    print("Wrong path!")
    exit()
audio_files = os.listdir(f'{path}')
print(len(audio_files))
for k in audio_files:  # k = audio faili pxxx mapītē
    if 'p315' in k:
        os.remove(f'{path}{k}')
        print(f"Removed {path}{k}")
audio_files = os.listdir(f'{path}')
print(len(audio_files))

print("All done! =)")


