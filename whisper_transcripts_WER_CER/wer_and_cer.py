import os
from tqdm import tqdm
import re
import time
DEBUGGING = False

def append_CER(true_ar, gen_ar,true_ts_word, gen_ts_word, error_type):
    true_ts_char = [*true_ts_word]
    gen_ts_char = [*gen_ts_word]
    #TODO if no word error
    if error_type == 'sub':
        #TODO
        for i in range(max(len(true_ts_char), len(gen_ts_char))):
            if i < min(len(true_ts_char), len(gen_ts_char)):
                true_ar.append(true_ts_char[i])
                gen_ar.append(gen_ts_char[i])
            elif i >= len(true_ts_char):
                true_ar.append('')
                gen_ar.append(gen_ts_char[i])
            elif i >= len(gen_ts_char):
                true_ar.append(true_ts_char[i])
                gen_ar.append('')
        pass
    elif error_type == 'del':
        for i in range(len(true_ts_char)):
            true_ar.append(true_ts_char[i])
            gen_ar.append('')
    elif error_type == 'ins':
        for i in range(len(gen_ts_char)):
            true_ar.append('')
            gen_ar.append(gen_ts_char[i])
    elif error_type == 'no_error':
        if len(true_ts_char) != len(gen_ts_char):
            print("Something is not right")
        for i in range(len(true_ts_char)):
            true_ar.append(true_ts_char[i])
            gen_ar.append(gen_ts_char[i])

    return true_ar, gen_ar

def calculate_WER(true_ts, generated_ts):
    if true_ts == generated_ts or true_ts.lower() == generated_ts.lower():
        return 0, 0
    if generated_ts == '' or generated_ts is None:
        return 1, 1
    true_tx_punc = true_ts.lower().split(" ")
    generated_tx_punc = generated_ts.lower().strip().split(" ")
    true_tx = []
    generated_tx = []
    for i in true_tx_punc:
        if '-' in i:
            true_tx.append(i.split('-')[0])
            if i.split('-')[1]:
                true_tx.append(re.split('[?.,!]', i.split('-')[1])[0])
        else:
            true_tx.append(re.split('[?.,!]', i)[0])
    for i in generated_tx_punc:
        if '-' in i:
            generated_tx.append(i.split('-')[0])
            if i.split('-')[1]:
                generated_tx.append(re.split('[?.,!]', i.split('-')[1])[0])
        else:
            generated_tx.append(re.split('[?.,!]', i)[0])
    sub_error = 0
    ins_error = 0
    del_error = 0
    correct_words = 0
    sub_error_1 = 0
    ins_error_1 = 0
    del_error_1 = 0
    true_tx_ar = []
    gen_text_ar = []
    j = 0
    k = 0
    start_time = time.time()
    try:

        while False in [b in true_tx_ar for b in true_tx] or False in [n in gen_text_ar for n in generated_tx]:
            process_time = time.time() - start_time
            if process_time > 5 and not DEBUGGING:
                print("ERROR")
                print("True: ", true_ts)
                print("Generated:", generated_ts)
                print("True arr:", true_tx_ar)
                print("Gen arr: ", gen_text_ar)
                return 1.01, 1.01
            # if length of True transcript is reached => insertion error for generated
            if k >= len(true_tx) and j < len(generated_tx):
                for i in range(len(generated_tx) - j):
                    true_tx_ar.append('')
                    gen_text_ar.append(generated_tx[j+i])
                    ins_error_1 += 1
                    # print("This used 5")
            # if length of Generated transcript is reached => deletion error for true
            elif j >= len(generated_tx) and k < len(true_tx):
                for i in range(len(true_tx) - k):
                    true_tx_ar.append(true_tx[i+k])
                    gen_text_ar.append('')
                    del_error_1 += 1
                    # print("This used 56")
            # If both words at the same position are the same => no error
            elif true_tx[k] == generated_tx[j]:
                true_tx_ar.append(true_tx[k])
                gen_text_ar.append(generated_tx[j])
                k += 1
                j += 1

            elif true_tx[k] in generated_tx:
                # if true_tx[k] == generated_tx[j-1]:
                #     pass
                if j + 1 < len(generated_tx):
                    if true_tx[k] == generated_tx[j+1]:
                        true_tx_ar.append('')
                        gen_text_ar.append(generated_tx[j])
                        ins_error_1 += 1
                        j += 1
                        continue

                if j + 1 < len(generated_tx) and k + 1 < len(true_tx):
                    if true_tx[k + 1] == generated_tx[j + 1]:
                        true_tx_ar.append(true_tx[k])
                        gen_text_ar.append(generated_tx[j])
                        sub_error_1 += 1
                        j += 1
                        k += 1
                        continue
                    if j < generated_tx.index(true_tx[k]) + 1 < len(generated_tx) and len(generated_tx) - len(true_tx) >= generated_tx.index(true_tx[k]) - k :
                        if true_tx[k + 1] == generated_tx[generated_tx.index(true_tx[k]) + 1]:
                            for i in range(generated_tx.index(true_tx[k]) - j):
                                true_tx_ar.append('')
                                gen_text_ar.append(generated_tx[j])
                                ins_error_1 += 1
                                j += 1
                            continue
                    if k + 2 < len(true_tx):
                        if true_tx[k + 2] == generated_tx[j + 1]:
                            true_tx_ar.append(true_tx[k])
                            true_tx_ar.append(true_tx[k+1])
                            gen_text_ar.append(generated_tx[j])
                            gen_text_ar.append('')
                            sub_error_1 += 1
                            del_error_1 += 1
                            j += 1
                            k += 2
                            continue

                true_tx_ar.append(true_tx[k])
                gen_text_ar.append(generated_tx[j])
                sub_error_1 += 1
                k += 1
                j += 1
                continue

            elif true_tx[k] not in generated_tx:
                if j+1 < len(generated_tx) and k+1 < len(true_tx):
                    # checks if the next words are both the same, if yes => substitution error
                    if true_tx[k+1] == generated_tx[j+1]:
                        true_tx_ar.append(true_tx[k])
                        gen_text_ar.append(generated_tx[j])
                        sub_error_1 +=1
                        j+=1
                        k+=1
                        continue
                    if k + 2 < len(true_tx):
                        if true_tx[k + 2] == generated_tx[j + 1]:
                            true_tx_ar.append(true_tx[k])
                            true_tx_ar.append(true_tx[k+1])
                            gen_text_ar.append(generated_tx[j])
                            gen_text_ar.append('')
                            sub_error_1 += 1
                            del_error_1 += 1
                            j += 1
                            k += 2
                            continue
                    else:
                        true_tx_ar.append(true_tx[k])
                        gen_text_ar.append(generated_tx[j])
                        sub_error_1 += 1
                        k += 1
                        j += 1
                        continue
                true_tx_ar.append(true_tx[k])
                gen_text_ar.append(generated_tx[j])
                sub_error_1 += 1
                k += 1
                j += 1

        for i in range(len(true_tx_ar)):
            if true_tx_ar[i]==gen_text_ar[i]:
                correct_words += 1
                continue
            elif true_tx_ar[i]=='':
                ins_error +=1
            elif gen_text_ar[i]=='':
                del_error += 1
            else:
                sub_error += 1

        word_error_rate = (sub_error + ins_error + del_error) / len(true_tx)
        word_error_rate_normalised = (sub_error + ins_error + del_error) / (sub_error + ins_error + del_error + correct_words)
        word_error_rate_1 = (sub_error_1 + ins_error_1 + del_error_1) / len(true_tx_ar)
        if word_error_rate > 1:
            word_error_rate = 1
        # if word_error_rate > 0.5 or word_error_rate_1 > 0.5:
        #     print(f"HIGH WER! WER: {word_error_rate}, WER_!: {word_error_rate_1}")
        #     print("TRUE: \t", true_tx_ar)
        #     print("GEN:  \t", gen_text_ar)

        return word_error_rate, word_error_rate_normalised
        # return word_error_rate
    except:
        print("An error occured")
        return 1.01, 1.01


def calculate_CER(true_ts, generated_ts):
    if true_ts == generated_ts:
        return 0, 0
    if generated_ts == '' or generated_ts is None:
        return 1, 1
    true_tx_ch = true_ts.split(" ")
    true_tx = []
    generated_tx = []
    for i in true_tx_ch:
        if '-' in i:
            true_tx.append(i.split('-')[0])
            true_tx.append('-')
            true_tx.append(i.split('-')[1])
        else:
            true_tx.append(i)
    generated_tx_ch = generated_ts.strip().split(" ")
    for i in generated_tx_ch:
        if '-' in i:
            generated_tx.append(i.split('-')[0])
            generated_tx.append('-')
            generated_tx.append(i.split('-')[1])
        else:
            generated_tx.append(i)
    true_tx_ar = []
    gen_tx_ar = []
    true_ch_ar = []
    gen_ch_ar = []
    sub_error = 0
    ins_error = 0
    del_error = 0
    correct_char = 0
    sub_error_1 = 0
    ins_error_1 = 0
    del_error_1 = 0
    j = 0
    k = 0
    start_time = time.time()
    try:

        while False in [b in true_tx_ar for b in true_tx] or False in [n in gen_tx_ar for n in generated_tx]:
            process_time = time.time() - start_time
            if process_time > 5 and not DEBUGGING:
                print("ERROR")
                print("True: ", true_ts)
                print("Generated:", generated_ts)
                print("True arr:", true_tx_ar)
                print("Gen arr: ", gen_tx_ar)
                return 1.01, 1.01
            # if length of True transcript is reached => insertion error for generated
            if k >= len(true_tx) and j < len(generated_tx):
                for i in range(len(generated_tx) - j):
                    true_tx_ar.append('')
                    gen_tx_ar.append(generated_tx[j+i])
                    ins_error_1 += len([*generated_tx[j+i]])
                    true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar, true_ts_word='',
                                                       gen_ts_word=generated_tx[j+i], error_type='ins')
                    true_ch_ar.append(" ")
                    gen_ch_ar.append(" ")
                    # print("This used 5")
            # if length of Generated transcript is reached => deletion error for true
            elif j >= len(generated_tx) and k < len(true_tx):
                for i in range(len(true_tx) - k):
                    true_tx_ar.append(true_tx[i+k])
                    gen_tx_ar.append('')
                    del_error_1 += len([*true_tx[i+k]])
                    true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar, true_ts_word=true_tx[i+k],
                                                       gen_ts_word='', error_type='del')
                    true_ch_ar.append(" ")
                    gen_ch_ar.append(" ")
                    # print("This used 56")
            # If both words at the same position are the same => no error
            elif true_tx[k] == generated_tx[j]:
                true_tx_ar.append(true_tx[k])
                gen_tx_ar.append(generated_tx[j])
                true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar, true_ts_word=true_tx[k],
                                                   gen_ts_word=generated_tx[j], error_type='no_error')
                true_ch_ar.append(" ")
                gen_ch_ar.append(" ")
                k += 1
                j += 1

            elif true_tx[k] in generated_tx:
                # if true_tx[k] == generated_tx[j-1]:
                #     pass
                if j + 1 < len(generated_tx):
                    if true_tx[k] == generated_tx[j+1]:
                        true_tx_ar.append('')
                        gen_tx_ar.append(generated_tx[j])
                        ins_error_1 += len(generated_tx[j])
                        true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar, true_ts_word='',
                                                           gen_ts_word=generated_tx[j], error_type='ins')
                        true_ch_ar.append(" ")
                        gen_ch_ar.append(" ")
                        j += 1
                        continue

                if j + 1 < len(generated_tx) and k + 1 < len(true_tx):
                    if true_tx[k + 1] == generated_tx[j + 1]:
                        true_tx_ar.append(true_tx[k])
                        gen_tx_ar.append(generated_tx[j])
                        #TODO
                        sub_error_1 += max(len(true_tx[k]), len(generated_tx[j]))
                        true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar, true_ts_word=true_tx[k],
                                                           gen_ts_word=generated_tx[j], error_type='sub')
                        true_ch_ar.append(" ")
                        gen_ch_ar.append(" ")
                        j += 1
                        k += 1
                        continue
                    if j < generated_tx.index(true_tx[k]) + 1 < len(generated_tx) and len(generated_tx) - len(true_tx) >= generated_tx.index(true_tx[k]) - k :
                        if true_tx[k + 1] == generated_tx[generated_tx.index(true_tx[k]) + 1]:
                            for i in range(generated_tx.index(true_tx[k]) - j):
                                true_tx_ar.append('')
                                gen_tx_ar.append(generated_tx[j])
                                ins_error_1 += len(generated_tx[j])
                                true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar,
                                                                   true_ts_word='',
                                                                   gen_ts_word=generated_tx[j], error_type='ins')
                                true_ch_ar.append(" ")
                                gen_ch_ar.append(" ")
                                j += 1
                            continue
                    if k + 2 < len(true_tx):
                        if true_tx[k + 2] == generated_tx[j + 1]:
                            true_tx_ar.append(true_tx[k])
                            true_tx_ar.append(true_tx[k+1])
                            gen_tx_ar.append(generated_tx[j])
                            gen_tx_ar.append('')
                            sub_error_1 += max(len(true_tx[k]), len(generated_tx[j]))
                            del_error_1 += len(true_tx[k+1])
                            true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar,
                                                               true_ts_word=true_tx[k],
                                                               gen_ts_word=generated_tx[j], error_type='sub')
                            true_ch_ar.append(" ")
                            gen_ch_ar.append(" ")
                            true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar,
                                                               true_ts_word=true_tx[k+1],
                                                               gen_ts_word='', error_type='del')
                            true_ch_ar.append(" ")
                            gen_ch_ar.append(" ")
                            j += 1
                            k += 2
                            continue

                true_tx_ar.append(true_tx[k])
                gen_tx_ar.append(generated_tx[j])
                sub_error_1 += max(len(true_tx[k]), len(generated_tx[j]))
                true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar,
                                                   true_ts_word=true_tx[k],
                                                   gen_ts_word=generated_tx[j], error_type='sub')
                true_ch_ar.append(" ")
                gen_ch_ar.append(" ")
                k += 1
                j += 1
                continue

            elif true_tx[k] not in generated_tx:
                if j+1 < len(generated_tx) and k+1 < len(true_tx):
                    # checks if the next words are both the same, if yes => substitution error
                    if true_tx[k+1] == generated_tx[j+1]:
                        true_tx_ar.append(true_tx[k])
                        gen_tx_ar.append(generated_tx[j])
                        sub_error_1 += max(len(true_tx[k]), len(generated_tx[j]))
                        true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar,
                                                           true_ts_word=true_tx[k],
                                                           gen_ts_word=generated_tx[j], error_type='sub')
                        true_ch_ar.append(" ")
                        gen_ch_ar.append(" ")
                        j += 1
                        k += 1
                        continue
                    if k + 2 < len(true_tx):
                        if true_tx[k + 2] == generated_tx[j + 1]:
                            true_tx_ar.append(true_tx[k])
                            true_tx_ar.append(true_tx[k+1])
                            gen_tx_ar.append(generated_tx[j])
                            gen_tx_ar.append('')
                            sub_error_1 += max(len(true_tx[k]), len(generated_tx[j]))
                            del_error_1 += len(true_tx[k+1])
                            true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar,
                                                               true_ts_word=true_tx[k],
                                                               gen_ts_word=generated_tx[j], error_type='sub')
                            true_ch_ar.append(" ")
                            gen_ch_ar.append(" ")
                            true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar,
                                                               true_ts_word=true_tx[k+1],
                                                               gen_ts_word='', error_type='del')
                            true_ch_ar.append(" ")
                            gen_ch_ar.append(" ")
                            j += 1
                            k += 2
                            continue
                    else:
                        true_tx_ar.append(true_tx[k])
                        gen_tx_ar.append(generated_tx[j])
                        sub_error_1 += max(len(true_tx[k]), len(generated_tx[j]))
                        true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar,
                                                           true_ts_word=true_tx[k],
                                                           gen_ts_word=generated_tx[j], error_type='sub')
                        true_ch_ar.append(" ")
                        gen_ch_ar.append(" ")
                        k += 1
                        j += 1
                        continue
                true_tx_ar.append(true_tx[k])
                gen_tx_ar.append(generated_tx[j])
                sub_error_1 += max(len(true_tx[k]), len(generated_tx[j]))
                true_ch_ar, gen_ch_ar = append_CER(true_ar=true_ch_ar, gen_ar=gen_ch_ar,
                                                   true_ts_word=true_tx[k],
                                                   gen_ts_word=generated_tx[j], error_type='sub')
                true_ch_ar.append(" ")
                gen_ch_ar.append(" ")
                k += 1
                j += 1

        for i in range(len(true_ch_ar)):
            if true_ch_ar[i] == gen_ch_ar[i]:
                correct_char += 1
                continue
            elif true_ch_ar[i] == '':
                ins_error += 1
            elif gen_ch_ar[i] == '':
                del_error += 1
            else:
                sub_error += 1

        character_error_rate = (sub_error + ins_error + del_error) / len([*true_ts])
        character_error_rate_normalised = (sub_error + ins_error + del_error) / (sub_error + ins_error + del_error + correct_char)
        character_error_rate_1 = (sub_error_1 + ins_error_1 + del_error_1) / len(true_ch_ar)
        if character_error_rate > 1:
            character_error_rate = 1
        # if character_error_rate > 0.5 or character_error_rate_1 > 0.5:
        #     print(f"HIGH CER! CER: {character_error_rate}, CER_1: {character_error_rate_1}")
        #     print("TRUE: \t", true_ch_ar)
        #     print("GEN:  \t", gen_ch_ar)

        return character_error_rate, character_error_rate_normalised
        # return character_error_rate
    except:
        print("An error occured")
        return 1.01, 1.01


if __name__ == '__main__':
    WRITE_DATA = False
    real_str = 'When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.'
    predicted_str = ' Waana man looks for something beyond his reach, his friends that is he is looking for the pot of gold at the end of the rainbow'
    WER_123, WERN_123 = calculate_WER(real_str, predicted_str)
    CER_123, CERN_123 = calculate_CER(real_str, predicted_str)
    print(WER_123*100, "       ", WERN_123*100)
    print(CER_123*100, "       ", CERN_123*100)
    if WRITE_DATA:
        # path_to_generated_transcript_folder = '../p278_transcripts_VCTK_original__W_medium_no-fine-tuning.txt'
        path_to_generated_transcript_folder = '../results/whisper_transcripts/'
        # path_to_true_transcript = './FreeVC/data/txt/'
        path_to_true_transcript = '../../datasets/VCTK-Corpus-0.92/txt/'
        generated_transcripts = os.listdir(path_to_generated_transcript_folder)
        all_wer_and_cer = open(f"../wer_cer/all_wer_cer.txt", "w")
        total_files_nr = 0
        total_speakers_nr = 0
        total_WER = 0
        total_WER_1 = 0
        total_CER = 0
        total_CER_1 = 0
        for j in tqdm(generated_transcripts):
            if os.path.exists(f"{path_to_generated_transcript_folder}{j}"):
                generated_ts = open(f"{path_to_generated_transcript_folder}{j}", "r")
            else:
                print("No path for ", j)
                continue
            transcripts = generated_ts.readlines()
            total_speakers_nr += 1
            speaker_wer_and_cer = open(f"./wer_cer/{j.split('_')[0]}.txt", "w")
            nr_of_files = 0
            speaker_WER = 0
            speaker_WER_1 = 0
            speaker_CER = 0
            speaker_CER_1 = 0
            for i in tqdm(transcripts):


                file, generated_transcript = i.split('|')
                generated_transcript = generated_transcript.rstrip('\n').strip()
                speaker, audio_nr, _ = file.split('_')

                if os.path.exists(f"{path_to_true_transcript}{speaker}/{speaker}_{audio_nr}.txt"):
                    true_ts = open(f"{path_to_true_transcript}{speaker}/{speaker}_{audio_nr}.txt", "r")
                else:
                    print("No path for ", f"{path_to_true_transcript}{speaker}/{speaker}_{audio_nr}.txt")
                    break

                true_transcript = true_ts.readlines()[0].rstrip('\n').strip()

                WER = calculate_WER(true_ts=true_transcript, generated_ts=generated_transcript)
                CER = calculate_CER(true_ts=true_transcript, generated_ts=generated_transcript)
                speaker_wer_and_cer.write(f"{file}|{round(WER, 6)}|{round(CER, 6)}|\n")
                # print("WER: ", WER, "\t WER_1: ", WER_1)
                # print("CER: ", round(CER*100, 4), "\t\t\t\t CER_1: ", round(CER_1*100, 4))
                speaker_WER += WER
                speaker_CER += CER
                total_WER += WER
                total_CER += CER
                nr_of_files += 1
                total_files_nr +=1


                true_ts.close()
            speaker_wer_and_cer.close()
            # print("Total WER: ", total_WER/nr_of_files * 100, " Total WER_1: ", total_WER_1/nr_of_files * 100)
            # print("Total CER: ", total_CER / nr_of_files * 100, " Total CER_1: ", total_CER_1 / nr_of_files * 100)
            if nr_of_files != 0:
                all_wer_and_cer.write(f"{j.split('_')[0]}|{round(speaker_WER/nr_of_files, 6)}|{round(speaker_WER_1/nr_of_files, 6)}|{round(speaker_CER/nr_of_files, 6)}|{round(speaker_CER_1/nr_of_files, 6)}\n")

            generated_ts.close()
        # all_wer_and_cer.write(f"all_by_speakers|")
        if total_files_nr != 0:
            all_wer_and_cer.write(
                f"all|{round(total_WER / total_files_nr,6)}|{round(total_WER_1 / total_files_nr,6)}|{round(total_CER / total_files_nr,6)}|{round(total_CER_1 / total_files_nr,6)}\n")
        all_wer_and_cer.close()

