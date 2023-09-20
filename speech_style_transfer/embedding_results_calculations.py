import os
import time

import torchaudio
import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import scipy.spatial.distance
from speechbrain.pretrained import EncoderClassifier
from convert_FreeVC import convert_audio
start_time = time.time()

checkpoint_path = "../pretrained_models/EncoderClassifier"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


DEVICE = "cpu"
speech_brain = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE},
    # device=DEVICE,
    savedir=checkpoint_path
)


file_or = '../audio_examples/emb/p225_007_mic1.flac'
file_to = '../audio_examples/emb/p269_007_mic1.flac'
file_vc = '../audio_examples/emb/p225_004_mic1_to_p269.wav'
audio, path_vc = convert_audio(file_or, file_to)

a = 0
signal_or, fs_or =torchaudio.load(file_or)
embeddings_or = np.mean(np.array(speech_brain.encode_batch(signal_or)), 1)
signal_to, fs_to =torchaudio.load(file_to)
embeddings_to = np.mean(np.array(speech_brain.encode_batch(signal_to)), 1)
signal_vc, fs_vc =torchaudio.load(path_vc)
embeddings_vc = np.mean(np.array(speech_brain.encode_batch(signal_vc)), 1)

dist_cos_other = 1 + scipy.spatial.distance.cdist(embeddings_or, embeddings_to, metric='cosine')
dist_cos_conv = 1 + scipy.spatial.distance.cdist(embeddings_or, embeddings_vc, metric='cosine')
score_p = 0.5 * dist_cos_other + 2 - dist_cos_conv
print(dist_cos_other, dist_cos_conv, score_p[0][0])

file_to = '../audio_examples/emb/p225_007_mic1.flac'
file_or = '../audio_examples/emb/p269_007_mic1.flac'
audio, path_vc = convert_audio(file_or, file_to)

signal_or, fs_or =torchaudio.load(file_or)
embeddings_or = np.mean(np.array(speech_brain.encode_batch(signal_or)), 1)
signal_to, fs_to =torchaudio.load(file_to)
embeddings_to = np.mean(np.array(speech_brain.encode_batch(signal_to)), 1)
signal_vc, fs_vc =torchaudio.load(path_vc)
embeddings_vc = np.mean(np.array(speech_brain.encode_batch(signal_vc)), 1)

dist_cos_other = 1 + scipy.spatial.distance.cdist(embeddings_or, embeddings_to, metric='cosine')
dist_cos_conv = 1 + scipy.spatial.distance.cdist(embeddings_or, embeddings_vc, metric='cosine')
score_p = 0.5 * dist_cos_other + 2 - dist_cos_conv
print(dist_cos_other, dist_cos_conv, score_p[0][0])
end_time = time.time()
print(end_time-start_time)

# np_embs_each = speech_brain_1.encode_batch(torch.FloatTensor(np_y_sub_parts_each)).cpu().data.numpy().squeeze(axis=1)

