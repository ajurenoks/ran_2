import os
import shutil

# path_to_file ='../datasets/VCTK-Corpus-0.92/wav48_silence_trimmed'
# path_to_file = './DR-VCTK_Office1_OpenedWindow'
path_to_file = './wav48_silence_trimmed'
end_path_train = './train'
end_path_test = './test'
end_path_val = './val'
if not os.path.exists(end_path_train):
    os.makedirs(end_path_train)
if not os.path.exists(end_path_test):
    os.makedirs(end_path_test)
if not os.path.exists(end_path_val):
    os.makedirs(end_path_val)

train_text = open("train.txt", "r")
test_text = open("test.txt", "r")
val_text = open("val.txt", "r")
lines_train = train_text.readlines()
lines_test = test_text.readlines()
lines_val = val_text.readlines()
for i in lines_train:
    end_fold = i.rstrip('\n')
    shutil.move(f'{path_to_file}/{end_fold}', end_path_train)
print("Done with train")
for i in lines_test:
    end_fold = i.rstrip('\n')
    shutil.move(f'{path_to_file}/{end_fold}', end_path_test)
print("Done with test")
for i in lines_val:
    end_fold = i.rstrip('\n')
    shutil.move(f'{path_to_file}/{end_fold}', end_path_val)
print("Done with val")
train_text.close()
test_text.close()
val_text.close()
print("All done")

