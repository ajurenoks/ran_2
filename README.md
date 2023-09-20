# Rezultāti
Visi iegūtie rezultāti ir aplūkojami mapītē `Transcript_results`, kur visiem runātāju un atsevišķajam datu kopas modelim ir sava mapīte.

Rezultāti apkopotā viedā ir aplūkojami `RESULTS.xlsx`

# Datu kopa

Ielādējot datu kopu (VCTK), palaiž (./dataset_configurations):
- no datu kopas `wav48_silence_trimmed` izdzēš `p315` mapīti
- `split_data_train_test_val.py` nomainot `path_to_file` uz ceļu līdz `wav48_silence_trimmed` mapītei
- `move_train_test_val.py` nomainot  `path_to_file` uz ceļu līdz `wav48_silence_trimmed` mapītei
-  `delete_mic2.py`, nomainot `path` uz to vietu, kur atrodas datu kopas train/test/val mapītes
- `write_metadata.py`, nomainot `path` uz to vietu, kur atrodas datu kopas train/test/val mapītes (Tur pat arī jāatrodas `txt` mapītei). Atstāj `rewrite_metadata = True` un `user_finetuning = False` 

# Whisper trenēšana

Lai trenētu Whisper modeli, palaiž `whisper_finetune.py` (./whisper/), norādot ceļu uz datu kopu `path_ds`, un, ja trenē specifiskam runātājam, `tune_for_specific_user = True` un `specific_user` norāda specifisko runātāju uz kuru trenē, citādi `tune_for_specific_user = False`.  
- Izveido mapīti `./pretrained_models`, kurā saglabā Whisper modeli un  `whisper_model_size` norāda modeļa attiecīgo izmēru

Lai turpinātu apmācības procesu no checkpointa, nomaina `model_path` uz attiecīgā checkpointa vietu un nomaina `trainer.train(resume_from_checkpoint=model_path)`

# FreeVC pārveidošana
Palaiž `file_for_FreeVC.py` (./dataset_configurations/),kas sataisa failu (nepieciešams lai val.txt un val mapītē atrodas tikai mērķa runātājs un pārējie runātāji ir train/test)

Palaiž `convert.py` no tās mapītes, kurā atrodas FreeVC modelis, nomainot `txtpath` uz iepriekš definēto mapīti (pēc noklusējuma convert_to_pxxx.txt). Ieteikums nomainīt arī `outdir` uz freevc_pxxx (pxxx - ir mērķa runātāja apzīmējums).

Mapītē freevc_pxxx būs visi pārveidotie faili 

pārvieto arī mērķa runātāja mapīti uz `outdir` mapīti (freevc_pxxx)

Palaiž `write_metadata.py`, nomainot `user_finetuning = True` un `specific_user = 'pxxx'` (pxxx vietā ieliek mērķa runātāju). Ja nepieciešams, nomaina `path_to_audio` uz to mapīti, kurā saglabāti visi pārveidotie faili. 

# Transkriptu iegūšana

Trenētie modeļi tika glabāti mapītē `pretrained_models`, ar nosaukumiem `whisper-{whisper_model_size}-finetuned-{pxxx/VCTK}/{training_date}/{checkpoint}`, kur `{pxxx/VCTK}` ir atkarīgs vai ir `finetuned_on_user`.

Palaiž `whisper_transcripts_wer_cer.py` (./whisper_transcripts_WER_CER/), nomainot:
  - `finetuned_on_user` uz `True`, ja tiek iegūtu transkripti no mērķa runātāja failiem, `False`, ja no visas datu kopas
  - `specific_user = 'pxxx'` ja tiek iegūtu transkripti no mērķa runātāja failiem, norādot `pxxx` kā attiecīgo runātāju
  - `whisper_model_size` norādot attiecīgo Whisper modeļa izmēru
  - `training_date` nomaina uz modeļa trenēšanas datumu
  - `checkpoint_nr` nomaina uz checkponta numuru 
  - VAI vispārīgi nomaina `path_to_model` uz ceļu līdz modelim, kuru izmanto transkriptu iegūšanai


Palaiž `transcripts_debugged_wer_cer.py` (./whisper_transcripts_WER_CER/), lai iegūtu WER un CER rezultātus, nomainot `path_to_files` uz to vietu, kur ir saglabāti iepriekš palaistā koda faili un `files_to_debug` norādot mapītes, kuras izmantot CER un WER iegūšanai.

# WER un CER

Lai iegūtu atsevišķiem teikumiem WER un CER palaiž `wer_and_cer.py` (./whisper_transcripts_WER_CER/), nomainot `real_str` uz reālo transkriptu un `predicted_str` uz ģenerēto transkriptu.  


# Cits info

## Conda envs

See current enviornments: conda info --envs

Create a new env: conda create --name `env_name`

Activate environment: conda activate `env_name`

## Packages

- python 3.9 (?) at least that is what I did/use
- torch (conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia `https://pytorch.org/get-started/locally/`)
- evaluate (pip install evaluate)
- Whisper (pip install -U openai-whisper)
  - Install base.en model in folder pretrained_models (from: `https://huggingface.co/openai/whisper-base.en/tree/main`)
- transformers (pip install transformers)
- audioread (conda install -c conda-forge audioread)
- tqdm (conda install -c conda-forge tqdm)
- pandas (conda install -c anaconda pandas)
- ffmpeg (conda install -c conda-forge ffmpeg) 
- jpeg(?) (conda install -c conda-forge jpeg)
- pillow(?) (conda install -c anaconda pillow)
- numpy(?) (conda install -c anaconda numpy)
- soundfile (conda install -c conda-forge pysoundfile)
- librosa (install -c conda-forge librosa)
-  (pip install jiwer)
-  (pip install --upgrade accelerate)
- tensorboard (conda install -c conda-forge tensorboard)
- FreeVC (https://github.com/OlaWod/FreeVC/archive/refs/heads/main.zip)
  - under wavlm folder download WavLM Large model
    - https://github.com/microsoft/unilm/tree/master/wavlm
  - webrtcvad (conda install -c conda-forge webrtcvad)
  - create a folder checkpoints and add freevc.pth to it (https://huggingface.co/spaces/OlaWod/FreeVC/tree/main/checkpoints)
- datasets (conda install -c conda-forge datasets)

[//]: # (- DEMUCS DENOISER)

[//]: # (  - pip install sphfile)

[//]: # (  - pip install pystoi)

[//]: # (  - pip install pesq)





