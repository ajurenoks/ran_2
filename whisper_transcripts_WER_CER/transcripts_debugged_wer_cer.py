path_to_files= '../results/GPU_transcripts_wer_cer'
files_to_debug=[
                # f'{path_to_files}/VCTK-p254_base_en/17_04_checkpoint-5400',
                # f'{path_to_files}/VCTK-all_base_en/13_04_checkpoint-18300',
                # f'{path_to_files}/VCTK-p287_base_en/17_04_checkpoint-3500',
                # f'{path_to_files}/VCTK-p304_base_en/16_05_checkpoint-2500',
                # f'{path_to_files}/VCTK-p304_base_en/16_05_checkpoint-5400_as_usual',
                # f'{path_to_files}/VCTK-p317_base_en/18_04_checkpoint-6600',
                # f'{path_to_files}/VCTK-p363_base_en/24_04_checkpoint-8500',
                # f'{path_to_files}/VCTK-p363_base_en/22_05_checkpoint-2300',
                # f'{path_to_files}/VCTK-p363_base_en/22_05_checkpoint-14100_1e-7'
                # f'{path_to_files}/VCTK-p317_base_en/23_05_checkpoint-700',
                # f'{path_to_files}/VCTK-p287_base_en/24_05_checkpoint-1600',
                f'{path_to_files}/VCTK-p254_base_en/25_05_checkpoint-1100',
                f'{path_to_files}/VCTK-p254_base_en/25_05_checkpoint-1500',
]
from wer_and_cer import calculate_WER, calculate_CER
import csv
import os
from tqdm import tqdm


for file_dir in files_to_debug:
    all_files = f"{file_dir}/ALL_speakers.csv"
    speaker_total_results = {}
    if os.path.exists(all_files):
        all_info = open(all_files, 'r')
    else:
        print("HUH?")
    all_info.readline()
    for line in tqdm(all_info.readlines()):
        output_folder = f"../results/WER_CER/{file_dir.split('/')[-2]}/{file_dir.split('/')[-1]}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if line.replace("\n", "").split(",")[0][0] !='p':
            continue
        else:
            line_arr = line.replace("\n", "").split(",")
            speaker = line_arr[0] # speaker pxxx
            if speaker_total_results.get(speaker) is None:
                speaker_total_results[speaker] = {'wer': 0, 'wer_n': 0, 'cer': 0, 'cer_n': 0,
                                                  'files_count': 0}


            transcripts_speaker = open(f"{output_folder}/{speaker}.csv", 'w', newline='')
            writer_speaker = csv.writer(transcripts_speaker)
            header = ['name', 'wer', 'wer_n', 'cer',  'cer_n', 'transcription', 'prediction']
            writer_speaker.writerow(header)
            if os.path.exists(f"{file_dir}/{speaker}.csv"):
                speaker_info = csv.DictReader(open(f"{file_dir}/{speaker}.csv", encoding='utf-8'))
            else:
                print("HUH?")
            for line_speaker in speaker_info:

                a=0
                file_name = line_speaker['name']
                transcription = line_speaker['transcription']
                prediction = line_speaker['prediction']
                wer, wer_n = calculate_WER(true_ts=transcription, generated_ts=prediction)
                cer, cer_n = calculate_CER(true_ts=transcription, generated_ts=prediction)
                speaker_total_results[speaker]['wer'] += wer*100
                speaker_total_results[speaker]['wer_n'] += wer_n*100
                speaker_total_results[speaker]['cer'] += cer*100
                speaker_total_results[speaker]['cer_n'] += cer_n * 100
                speaker_total_results[speaker]['files_count'] += 1
                if wer> 1 or cer > 1:
                    print(file_name)
                data_i = [file_name, wer*100, wer_n*100, cer*100, cer_n*100, transcription, prediction]
                try:
                    writer_speaker.writerow(data_i)
                except:
                    print("WHOOPSIE AN ERROR")
                    print(data_i)
            transcripts_speaker.close()
    all_info.close()
    transcripts = open(f"{output_folder}/ALL_speakers.csv", 'w', newline='')
    writer_all = csv.writer(transcripts)
    header_all = ['name', 'wer', 'wer_n', 'cer', 'cer_n']
    writer_all.writerow(header_all)
    wer_total = 0
    wer_n_total = 0
    cer_total = 0
    cer_n_total = 0
    total_files = 0

    wer_speakers = 0
    wer_n_speakers = 0
    cer_speakers = 0
    cer_n_speakers = 0

    for i in speaker_total_results:
        if speaker_total_results[i]['files_count'] != 0:
            data_i = [i, speaker_total_results[i]['wer'] / speaker_total_results[i]['files_count'],
                      speaker_total_results[i]['wer_n'] / speaker_total_results[i]['files_count'],
                      speaker_total_results[i]['cer'] / speaker_total_results[i]['files_count'],
                      speaker_total_results[i]['cer_n'] / speaker_total_results[i]['files_count']]
            writer_all.writerow(data_i)
            wer_total += speaker_total_results[i]['wer']
            wer_n_total += speaker_total_results[i]['wer_n']
            cer_total += speaker_total_results[i]['cer']
            cer_n_total += speaker_total_results[i]['cer_n']
            total_files += speaker_total_results[i]['files_count']

            wer_speakers += speaker_total_results[i]['wer'] / speaker_total_results[i]['files_count']
            wer_n_speakers += speaker_total_results[i]['wer_n'] / speaker_total_results[i]['files_count']
            cer_speakers += speaker_total_results[i]['cer'] / speaker_total_results[i]['files_count']
            cer_n_speakers += speaker_total_results[i]['cer_n'] / speaker_total_results[i]['files_count']

        a = 0
    if total_files != 0:
        data_total_files = ['total_files_total_wer_cer', wer_total / total_files, wer_n_total / total_files,
                            cer_total / total_files, cer_n_total / total_files]
        writer_all.writerow(data_total_files)
    if len(speaker_total_results) != 0:
        speakers_no = len(speaker_total_results)
        data_total_speakers = ['total_by_speakers', wer_speakers / speakers_no, wer_n_speakers / speakers_no,
                               cer_speakers / speakers_no, cer_n_speakers / speakers_no]
        writer_all.writerow(data_total_speakers)
    transcripts.close()



