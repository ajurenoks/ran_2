import os
import argparse
import torch
import librosa
import torchaudio
import numpy as np
from scipy.spatial.distance import cdist
import scipy.spatial.distance
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import os
from speechbrain.pretrained import EncoderClassifier

import sys
# path to FreeVC repo on PC
if not torch.cuda.is_available():
    path_to_FreeVC = '../../SST_kodi/FreeVC-main'
else:
    path_to_FreeVC = '../FreeVC'
if not os.path.exists(path_to_FreeVC):
    print("Wrong path to FreeVC")
    exit()
if not torch.cuda.is_available():
    path_to_splits = 'D:/BD/datasets/VCTK-Corpus-0.92/'
else:
    path_to_splits = '../datasets/VCTK/'
if not os.path.exists(path_to_splits):
    print("Wrong path to data files")
    exit()
sys.path.insert(1, path_to_FreeVC)

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

if __name__ == "__main__":
    start_time = time.time()
    checkpoint_path = "../pretrained_models/EncoderClassifier"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = "cpu"
    speech_brain = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
        # device=DEVICE,
        savedir=checkpoint_path
    )

    hps = utils.get_hparams_from_file(f"{path_to_FreeVC}/configs/freevc.json")
    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(f"{path_to_FreeVC}/checkpoints/freevc.pth", net_g, None, True)

    print("Loading WavLM for content...")
    # cmodel = utils.get_cmodel(0)
    # cmodel = torch.load(f'{path_to_FreeVC}/wavlm/WavLM-Large.pt')
    checkpoint = torch.load(f'{path_to_FreeVC}/wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg)
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()

    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder(f'{path_to_FreeVC}/speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print("Processing text...")

    speaker_files_used = {}
    titles, srcs, tgts = [], [], []

    for split_folder in os.listdir(path_to_splits):
        if split_folder != 'test' and split_folder != 'train' and split_folder != 'val':
            continue
        else:
            path_to_speaker_folders_real = f"{path_to_splits}{split_folder}"
            for speaker_real in os.listdir(path_to_speaker_folders_real):
                path_to_speaker_real_files = f"{path_to_speaker_folders_real}/{speaker_real}"
                speaker_real_files = os.listdir(path_to_speaker_real_files)
                for split_folder_other in os.listdir(path_to_splits):
                    if split_folder_other != 'test' and split_folder_other != 'train' and split_folder_other != 'val':
                        continue
                    else:
                        path_to_speaker_folders_other = f"{path_to_splits}{split_folder_other}"
                        for speaker_other in os.listdir(path_to_speaker_folders_other):
                            if speaker_real == speaker_other:
                                continue
                            else:
                                path_to_speaker_other_files = f"{path_to_splits}{split_folder_other}/{speaker_other}"
                                speaker_other_files = os.listdir(path_to_speaker_other_files)
                                titles.append(f"{speaker_real}_to_{speaker_other}")
                                if not speaker_files_used.get(speaker_real):
                                    speaker_files_used[speaker_real] = 1
                                    speaker_file_nr = 1
                                else:
                                    if speaker_files_used.get(speaker_real) + 1 >= len(speaker_real_files):
                                        speaker_file_nr = 1
                                        speaker_files_used[speaker_real] = 1
                                    else:
                                        speaker_file_nr = speaker_files_used.get(speaker_real) + 1
                                        speaker_files_used[speaker_real] = speaker_file_nr
                                if speaker_real_files[speaker_file_nr] != 'metadata.csv':
                                    pass
                                else:
                                    if speaker_file_nr < len(speaker_real_files) - 3:
                                        speaker_file_nr += 1
                                    else:
                                        speaker_file_nr = 1
                                srcs.append(f"{path_to_speaker_real_files}/{speaker_real_files[speaker_file_nr]}")
                                if not speaker_files_used.get(speaker_other):
                                    speaker_files_used[speaker_other] = 1
                                    speaker_file_nr = 1
                                else:
                                    if speaker_files_used.get(speaker_other) + 1 >= len(speaker_other_files):
                                        speaker_file_nr = 1
                                        speaker_files_used[speaker_other] = 1
                                    else:
                                        speaker_file_nr = speaker_files_used.get(speaker_other) + 1
                                        speaker_files_used[speaker_other] = speaker_file_nr
                                if speaker_other_files[speaker_file_nr] != 'metadata.csv':
                                    pass
                                else:
                                    if speaker_file_nr < len(speaker_other_files) - 3:
                                        speaker_file_nr += 1
                                    else:
                                        speaker_file_nr = 1
                                tgts.append(f"{path_to_speaker_other_files}/{speaker_other_files[speaker_file_nr]}")

    print("Synthesizing...")
    results_file = open('embedding_results.txt', 'w', encoding='utf-8')
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            try:
                title, src, tgt = line
                # tgt
                wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
                wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
                if hps.model.use_spk:
                    g_tgt = smodel.embed_utterance(wav_tgt)
                    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0)
                else:
                    wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0)
                    mel_tgt = mel_spectrogram_torch(
                        wav_tgt,
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax
                    )
                # src
                wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
                wav_src = torch.from_numpy(wav_src).unsqueeze(0)
                c = utils.get_content(cmodel, wav_src)

                if hps.model.use_spk:
                    audio = net_g.infer(c, g=g_tgt)
                else:
                    audio = net_g.infer(c, mel=mel_tgt)
                audio = audio[0][0].data.cpu().float().numpy()
                write("test.wav", hps.data.sampling_rate, audio)

                signal_or, fs_or = torchaudio.load(src)
                embeddings_or = np.mean(np.array(speech_brain.encode_batch(signal_or).to('cpu')), 1)
                signal_to, fs_to = torchaudio.load(tgt)
                embeddings_to = np.mean(np.array(speech_brain.encode_batch(signal_to).to('cpu')), 1)
                signal_vc, fs_vc = torchaudio.load('test.wav')
                embeddings_vc = np.mean(np.array(speech_brain.encode_batch(signal_vc).to('cpu')), 1)

                dist_cos_other = 1 + scipy.spatial.distance.cdist(embeddings_or, embeddings_to, metric='cosine')
                dist_cos_conv = 1 + scipy.spatial.distance.cdist(embeddings_or, embeddings_vc, metric='cosine')
                score_p = 0.5 * dist_cos_other + 2 - dist_cos_conv
                print(f"\nConversion {title}, dist_cos_other: {dist_cos_other}, dist_cos_conv: {dist_cos_conv}, score_p: {score_p[0][0]}")
                results_file.write(f"{title}|{dist_cos_other[0][0]}|{dist_cos_conv[0][0]}|{score_p[0][0]}\n")
            except:
                print("ERROR!!!")
                print(f"SOURCE: {src}, TARGET: {tgt}, FILE NAME: {title}" )
    results_file.close()

