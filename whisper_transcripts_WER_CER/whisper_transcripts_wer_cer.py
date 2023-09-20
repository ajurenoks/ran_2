from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
import os
import csv
from wer_and_cer import calculate_WER, calculate_CER
from tqdm import tqdm

finetuned_on_user = True
specific_user = 'p304'
user_nr = specific_user[1:]
whisper_model_size = 'base_en'
training_date = '16_05'
checkpoint_nr = 2_500
checkpoint = f"checkpoint-{checkpoint_nr}"
if not finetuned_on_user:
    whisper_model_dir = f'{whisper_model_size}-finetuned-VCTK' #/checkpoint-{checkpoint_nr}'
    # whisper_model_dir = whisper_model_size
    output_dir = f"./whisper_transcripts/VCTK-all_{whisper_model_size}/{training_date}_{checkpoint}"
else:
    whisper_model_dir = f'{whisper_model_size}-finetuned-{specific_user}/{training_date}/{checkpoint}'
    # whisper_model_dir = 'finetuned-p287-17_04'
    output_dir = f"./whisper_transcripts/VCTK-{specific_user}_{whisper_model_size}/{training_date}_{checkpoint}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


if torch.cuda.is_available():
    if not finetuned_on_user:
        path_ds = '../dataset/'
    else:
        path_ds = f'../FreeVC/FreeVC-main/output/freevc_{specific_user}/'
else:
    path_ds = '../../datasets/VCTK-Corpus-0.92/'
# VCTK -> test/train/val
# test/train/val -> pxxx
# pxxx -> audio faili

if not os.path.exists(path_ds):
    print("Wrong path! to dataset")
    exit()
dataset_full = DatasetDict()

if not finetuned_on_user:
    train_test_val_folders = os.listdir(path_ds)

    for i in train_test_val_folders:  # i = test/train/val
        if i == 'test' or i == 'train' or i == 'val':
            speaker_folders = os.listdir(f'{path_ds}{i}')
        else:
            print(f"{i} Not train/test/val")
            continue
        print(f"Transcripts started for {i} folder")
        for j in tqdm(speaker_folders):  # j = pxxx
            if j == 'p315':
                print("Text files for p315 do not exist, continue")
                continue
            if not os.path.exists(f'{path_ds}{i}/{j}'):
                print("Wrong audio path!")
                exit()
            # audio_files = os.listdir(f'{path_ds}{i}/{j}')
            dataset_x = load_dataset(f'{path_ds}{i}/{j}')
            dataset_x = dataset_x.cast_column("audio", Audio(sampling_rate=16000))
            if not dataset_full.get("train"):
                dataset_full["train"] = dataset_x["train"]
            else:
                dataset_full["train"] = concatenate_datasets([dataset_full["train"], dataset_x["train"]])

            # dataset_full = concatenate_datasets([dataset_full, dataset_x])
else:
    dataset_x = load_dataset(f'{path_ds}')
    dataset_x = dataset_x.cast_column("audio", Audio(sampling_rate=16000))
    dataset_full["train"] = dataset_x["train"]


path_to_model = f"./pretrained_models/whisper-{whisper_model_dir}"
if not os.path.exists(path_to_model):
    print("Wrong path to model!")
    print(path_to_model)
    exit()
processor = WhisperProcessor.from_pretrained(path_to_model,
                                             local_files_only=True, language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(path_to_model,
                                                        local_files_only=True)
if torch.cuda.is_available():
    model = model.to("cuda")

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    # batch["reference"] = processor.tokenizer._normalize(batch['transcription'])
    # batch["reference"] = batch['transcription']

    with torch.no_grad():
        if torch.cuda.is_available():
            predicted_ids = model.generate(input_features.to("cuda"))[0]
        else:
            predicted_ids = model.generate(input_features)[0]  # .to("cuda")

    # transcription = processor.decode(predicted_ids)
    transcription = processor.decode(predicted_ids, skip_special_tokens=True)
    # batch["prediction"] = processor.tokenizer._normalize(transcription)
    batch["prediction"] = transcription
    return batch
# result = vctk_ds_p226['train'].select([1, 2, 3, 4, 5]).map(map_to_pred)
# result = dataset_full['train'].select([1, 40692, 8933, 8934, 24895, 69, 420, 46, 935, 139, 358, 682, 2489, 1893]).map(map_to_pred)
result = dataset_full['train'].map(map_to_pred)

wer = load("wer")
cer = load("cer")
# a=0
# print(result["reference"])
# print(result["prediction"])


transcripts = open(f"{output_dir}/ALL_speakers.csv", 'w', newline='')
writer_all = csv.writer(transcripts)
header_all = ['name', 'wer_ev', 'cer_ev', 'wer_my', 'cer_my']
header = ['name', 'transcription', 'prediction', 'ev_wer', 'ev_cer', 'my_wer', 'my_cer']
writer_all.writerow(header_all)
speaker_total_results = {}
for i in tqdm(result):
    # i['speaker_file'] = pxxx_yyy
    # i['speaker_file'].split('_')[0] = pxxx
    # i['prediction'] - ģenerētais teksts
    speaker = i['speaker_file'].split('_')[0]
    transcript_i_file = f"{output_dir}/{speaker}.csv"
    if os.path.exists(transcript_i_file):
        transcripts_i = open(transcript_i_file, 'a', newline='')
        writer_i = csv.writer(transcripts_i)
    else:
        transcripts_i = open(transcript_i_file, 'w', newline='')
        writer_i = csv.writer(transcripts_i)
        writer_i.writerow(header)
    if speaker_total_results.get(speaker) is None:
        speaker_total_results[speaker] = {'wer_i': 0, 'cer_i': 0, 'wer_i_my': 0, 'cer_i_my': 0, 'files_count': 0}

    wer_i = 100 * wer.compute(references=[i["transcription"]], predictions=[i["prediction"]])
    if wer_i > 100:
        wer_i = 100
    cer_i = 100 * cer.compute(references=[i["transcription"]], predictions=[i["prediction"]])
    if cer_i > 100:
        cer_i = 100

    wer_i_my = 100 * calculate_WER(true_ts=i["transcription"], generated_ts=i["prediction"])
    cer_i_my = 100 * calculate_CER(true_ts=i["transcription"], generated_ts=i["prediction"])
    if wer_i_my >= 101 or cer_i_my >= 101:
        print("CER or WER >= 101 on ", i['speaker_file'])
        if wer_i_my > 100:
            wer_i_my = 100
        if cer_i_my > 100:
            cer_i_my = 100
    speaker_total_results[speaker]['wer_i'] += wer_i
    speaker_total_results[speaker]['cer_i'] += cer_i
    speaker_total_results[speaker]['wer_i_my'] += wer_i_my
    speaker_total_results[speaker]['cer_i_my'] += cer_i_my
    speaker_total_results[speaker]['files_count'] += 1

    data_i = [i["speaker_file"], i["transcription"], i["prediction"], wer_i, cer_i, wer_i_my, cer_i_my]
    try:
        writer_i.writerow(data_i)
    except:
        print("WHOOPSIE AN ERROR")
        print(data_i)
    transcripts_i.close()


    # print(100 * wer.compute(references=[i["transcription"]], predictions=[i["prediction"]]))
total_wer = 100 * wer.compute(references=result["transcription"], predictions=result["prediction"])
total_cer = 100 * cer.compute(references=result["transcription"], predictions=result["prediction"])
data_all = ['all_wer_cer', total_wer, total_cer, -1, -1]
writer_all.writerow(data_all)
wer_i_total = 0
wer_i_my_total = 0
cer_i_total = 0
cer_i_my_total = 0
total_files = 0

wer_i_speakers = 0
wer_i_my_speakers = 0
cer_i_speakers = 0
cer_i_my_speakers = 0

for i in speaker_total_results:
    if speaker_total_results[i]['files_count'] != 0:
        data_i = [i,  speaker_total_results[i]['wer_i']/speaker_total_results[i]['files_count'],
                  speaker_total_results[i]['cer_i']/speaker_total_results[i]['files_count'],
                  speaker_total_results[i]['wer_i_my']/speaker_total_results[i]['files_count'],
                  speaker_total_results[i]['cer_i_my']/speaker_total_results[i]['files_count']]
        writer_all.writerow(data_i)
        wer_i_total += speaker_total_results[i]['wer_i']
        wer_i_my_total += speaker_total_results[i]['cer_i']
        cer_i_total += speaker_total_results[i]['wer_i_my']
        cer_i_my_total += speaker_total_results[i]['cer_i_my']
        total_files += speaker_total_results[i]['files_count']

        wer_i_speakers += speaker_total_results[i]['wer_i']/speaker_total_results[i]['files_count']
        wer_i_my_speakers += speaker_total_results[i]['cer_i']/speaker_total_results[i]['files_count']
        cer_i_speakers += speaker_total_results[i]['wer_i_my']/speaker_total_results[i]['files_count']
        cer_i_my_speakers += speaker_total_results[i]['cer_i_my']/speaker_total_results[i]['files_count']




    a=0
if total_files != 0:
    data_total_files = ['total_files_total_wer_cer', wer_i_total/total_files, cer_i_total/total_files,
                        wer_i_my_total/total_files, cer_i_my_total/total_files]
    writer_all.writerow(data_total_files)
if len(speaker_total_results) != 0:
    speakers_no = len(speaker_total_results)
    data_total_speakers = ['total_by_speakers', wer_i_speakers/speakers_no, wer_i_my_speakers/speakers_no,
                           cer_i_speakers/speakers_no, cer_i_my_speakers/speakers_no]
    writer_all.writerow(data_total_speakers)
transcripts.close()
# print(100 * wer.compute(references=result["transcription"], predictions=result["prediction"]))
# print(100 * cer.compute(references=result["transcription"], predictions=result["prediction"]))