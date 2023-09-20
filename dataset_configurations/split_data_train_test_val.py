import os
import shutil

import pandas

path_to_file = './wav48_silence_trimmed'
files = os.listdir(path_to_file)
files.remove(files[0])
files.remove(files[-1])
train = []
# train_text = open("train.txt", "x")
train_text = open("train.txt", "w")
test = []
# test_text = open("test.txt", "x")
test_text = open("test.txt", "w")
val = []
# val_text = open("val.txt", "x")
val_text = open("val.txt", "w")

for i in range(len(files)):
    if i % 21 == 0 and i != 0:
        val.append(files[i])
        val_text.write(f'{files[i]}\n')
    elif i % 5 == 0:
        test.append(files[i])
        test_text.write(f'{files[i]}\n')
    else:
        train.append(files[i])
        train_text.write(f'{files[i]}\n')

print("Train size: ", len(train), " Data: ", train)
print("Test size: ", len(test), " Data: ", test)
print("Val size: ", len(val), " Data: ", val)

end_path_train = '../datasets/VCTK-Corpus-0.92/train'
end_path_test = '../datasets/VCTK-Corpus-0.92/test'
end_path_val = '../datasets/VCTK-Corpus-0.92/val'
if not os.path.exists(end_path_train):
    os.makedirs(end_path_train)
if not os.path.exists(end_path_test):
    os.makedirs(end_path_test)
if not os.path.exists(end_path_val):
    os.makedirs(end_path_val)



# for i in files:
#     if i in val:
#         shutil.move(f'{path_to_file}/{i}', end_path_val)
#     elif i in test:
#         shutil.move(f'{path_to_file}/{i}', end_path_test)
#     elif i in train:
#         shutil.move(f'{path_to_file}/{i}', end_path_train)
#     else:
#         print("Where dis: ", i)

train_text.close()
test_text.close()
val_text.close()