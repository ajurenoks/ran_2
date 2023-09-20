import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import sys
# path to FreeVC repo on PC
path_to_FreeVC = '../../SST_kodi/FreeVC-main'
sys.path.insert(1, path_to_FreeVC)

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

# class VoiceConversion:
#     def __init__(self):
#         pass
#     @staticmethod
def convert_audio(original_audio_path, target_audio_path):


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
    titles, srcs, tgts = [], [], []
    titles.append("test_file")
    srcs.append(original_audio_path)
    tgts.append(target_audio_path)
    # with open(args.txtpath, "r") as f:
    #     for rawline in f.readlines():
    #         title, src, tgt = rawline.strip().split("|")
    #         titles.append(title)
    #         srcs.append(src)
    #         tgts.append(tgt)

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
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
            write(os.path.join('../audio_examples/emb/', f"test.wav"), hps.data.sampling_rate, audio)
            path = '../audio_examples/emb/test.wav'
            return audio, path
            # if args.use_timestamp:
            #     timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
            #     write(os.path.join(args.outdir, "{}.wav".format(timestamp + "_" + title)), hps.data.sampling_rate,
            #           audio)
            # else:
            #     write(os.path.join(args.outdir, f"{title}.wav"), hps.data.sampling_rate, audio)

