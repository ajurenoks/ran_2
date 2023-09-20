import whisper
import os
import csv
from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets, Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

# whisper_model_size = 'base'
# feature_extractor = WhisperFeatureExtractor.from_pretrained(f"../pretrained_models/whisper-{whisper_model_size}",
#                                                             local_files_only=True)
# tokenizer = WhisperTokenizer.from_pretrained(f"../pretrained_models/whisper-{whisper_model_size}",
#                                              local_files_only=True, language="English", task="transcribe")
# processor = WhisperProcessor.from_pretrained(f"../pretrained_models/whisper-{whisper_model_size}",
#                                              local_files_only=True, language="English", task="transcribe")
# model = WhisperForConditionalGeneration.from_pretrained(f"../pretrained_models/whisper-{whisper_model_size}",
#                                                         local_files_only=True)
model = whisper.load_model("medium")
# path = './audio_examples'
# path = './datasets/VCTK/'
path = '../../datasets/VCTK-Corpus-0.92/'
# VCTK -> test/train/val
# test/train/val -> pxxx
# pxxx -> audio faili
if not os.path.exists(path):
    print("Wrong path!")
    exit()
train_test_val_folders = os.listdir(path)

for i in train_test_val_folders:  # i = test/train/val
    if i == 'test' or i == 'train' or i == 'val':
        speaker_folders = os.listdir(f'{path}{i}')
    else:
        print(f"{i} Not train/test/val")
        continue
    print(f"Transcripts started for {i} folder")
    for j in speaker_folders:  # j = pxxx
        if j == 'p315':
            print("Text files for p315 do not exist, continue")
            continue
        if not os.path.exists(f'{path}{i}/{j}'):
            print("Wrong audio path!")
            exit()
        audio_files = os.listdir(f'{path}{i}/{j}')

        transcripts = open(f"./whisper_transcripts/{j}_transcripts_VCTK_original_W_medium_no-fine-tuning.csv", 'w', newline='')
        writer = csv.writer(transcripts)
        header = ['file_name', 'transcription']
        writer.writerow(header)

        for k in audio_files:  # k = audio faili pxxx mapītē
            if k.split('.')[0].split('_')[-1] == 'mic1':
                result = model.transcribe(audio=f'{path}{i}/{j}/{k}', language='en')
                data = [k, result]
                writer.writerow(data)
                # transcripts.write(f'{k}|{result["text"]}\n')
                print(i, ' result: ', result["text"])

        transcripts.close()
        print(f"transcripts done for speaker {j} ")
print("All done! =)")
