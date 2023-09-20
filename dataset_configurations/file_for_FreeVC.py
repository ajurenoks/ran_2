# title2|DUMMY/p226/p226_002.wav|DUMMY/p225/p225_001.wav
# name | train_test| val
import os
# DR-VCTK -> VCTK
# val_data = '../val.txt'
# train_data = '../train.txt'
# test_data = '../test.txt'
val_data = '../../dataset/val.txt'
train_data = '../../dataset/train.txt'
test_data = '../../dataset/test.txt'
end_path_train = '../../dataset/train'
end_path_test = '../../dataset/test'
end_path_val = '../../dataset/val'
# path_to_splits = '../../datasets/VCTK-Corpus-0.92/'
# end_path_train = f'{path_to_splits}train'
# end_path_test = f'{path_to_splits}test'
# end_path_val = f'{path_to_splits}val'
# path = '../../datasets/VCTK-Corpus-0.92/wav48_silence_trimmed'
if not os.path.exists(val_data):
    print("Wrong Val path")
    exit()
if not os.path.exists(train_data):
    print("Wrong Train path")
    exit()
if not os.path.exists(test_data):
    print("Wrong Test path")
    exit()
if not os.path.exists(end_path_val):
    print("Wrong Val file path")
    exit()
if not os.path.exists(end_path_train):
    print("Wrong Train file path")
    exit()
if not os.path.exists(end_path_test):
    print("Wrong Test file path")
    exit()
train_text = open(train_data, "r")
test_text = open(test_data, "r")
val_text = open(val_data, "r")
lines_train = train_text.readlines()
lines_test = test_text.readlines()
lines_val = val_text.readlines()
for i in lines_val:
    val_fold = i.rstrip('\n')
    convert_file = open(f"convert_to_{val_fold}.txt", "w")
    print("created convert file: ", f"convert_to_{val_fold}.txt")
    files_in_val = os.listdir(f'{end_path_val}/{val_fold}')
    for j in lines_train:
        train_fold = j.rstrip('\n')
        # print(f"going through {train_fold} to {val_fold}")
        files_in_train = os.listdir(f'{end_path_train}/{train_fold}')
        files_in_train_mic1 = []
        files_in_train_mic2 = []
        for i in files_in_train:
            if i.split('.')[0].split('_')[-1] == 'mic1':
                files_in_train_mic1.append(i)
            elif i.split('.')[0].split('_')[-1] == 'mic2':
                files_in_train_mic2.append(i)
            else:
                print("No mic1 or mic2 file: ", i)
        for k in range(len(files_in_train_mic1)):
            if k < len(files_in_val):
                l= k
            else:
                l = k - len(files_in_val)
            if k > 2*len(files_in_val):
                l = k - 2*len(files_in_val)
            if files_in_val[l] == 'metadata.csv':
                if l < len(files_in_val) - 5:
                    l += 1
                else:
                    l = 1
            convert_file.write(f'{files_in_train_mic1[k].split(".", 1)[0]}_to_{val_fold}|{end_path_train}/{train_fold}/{files_in_train_mic1[k]}|{end_path_val}/{val_fold}/{files_in_val[l]}\n')
    for j in lines_test:
        test_fold = j.rstrip('\n')
        # print(f"going through {test_fold} to {val_fold}")
        files_in_test = os.listdir(f'{end_path_test}/{test_fold}')
        files_in_test_mic1 = []
        files_in_test_mic2 = []
        for i in files_in_test:
            if i.split('.')[0].split('_')[-1] == 'mic1':
                files_in_test_mic1.append(i)
            elif i.split('.')[0].split('_')[-1] == 'mic2':
                files_in_test_mic2.append(i)
            else:
                print("No mic1 or mic2 file: ", i)
        for k in range(len(files_in_test_mic1)):
            if k < len(files_in_val):
                l = k
            else:
                l = k - len(files_in_val)
            if k > 2 * len(files_in_val):
                l = k - 2 * len(files_in_val)
            if files_in_val[l] == 'metadata.csv':
                if l < len(files_in_val) - 5:
                    l += 1
                else:
                    l = 1
            convert_file.write(
                f'{files_in_test_mic1[k].split(".", 1)[0]}_to_{val_fold}|{end_path_test}/{test_fold}/{files_in_test_mic1[k]}|{end_path_val}/{val_fold}/{files_in_val[l]}\n')
    print(f"done with file: convert_to_{val_fold}.txt")

    # iet cauri test un train

    convert_file.close()
train_text.close()
test_text.close()
val_text.close()
print("All done")
