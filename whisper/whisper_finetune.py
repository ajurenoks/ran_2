import argparse

import logging
import random
import string
import os
import time

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets, Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

# if torch.cuda.is_available():
#     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#     torch.cuda.set_device(0)
#     torch.cuda.set_device(1)
#     torch.distributed.init_process_group('nccl', rank=0, world_size=1)

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


if __name__ == "__main__":
    # path = '../datasets/VCTK-Corpus-0.92/test/p225'
    if torch.cuda.is_available():
        path_ds = '../dataset/' #GPU
    else:
        print("NO CUDA!")
        path_ds = '../../datasets/VCTK-Corpus-0.92/' #LOCAL
    tune_for_specific_user = False
    specific_user = 'p304'
    whisper_model_size = 'base_en'
    if not os.path.exists(path_ds):
        print("Wrong path1!")
        exit()
    train_test_val_folders = os.listdir(path_ds)
    #TODO
    current_time = time.localtime()
    formatted_time = "{day:02d}_{month:02d}_{hour:02d}_{minute:02d}".format(
        day=current_time.tm_mday,
        month=current_time.tm_mon,
        hour=current_time.tm_hour,
        minute=current_time.tm_min,
    )

    parser = argparse.ArgumentParser(description=f"Finetune whisper {whisper_model_size} model for VCTK dataset.")
    parser.add_argument("--dataset_name", type=str, default='audiofolder')
    parser.add_argument("--model_size", type=str, default=f"{whisper_model_size}")
    if tune_for_specific_user:
        parser.add_argument("--output_dir", type=str, default=f'./results/whisper-{whisper_model_size}-finetune-{specific_user}-{formatted_time}')
        print("FINETUNING FOR ", specific_user)
    else:
        parser.add_argument("--output_dir", type=str, default=f'./results/whisper-{whisper_model_size}-finetune-VCTK-{formatted_time}')
        print("FINETUNING FOR VCTK")

    for i in train_test_val_folders:  # i = test/train/val
        if i == 'test' or i == 'train' or i == 'val':
            speaker_folders = os.listdir(f'{path_ds}{i}')
        else:
            print(f"{i} Not train/test/val")
            continue
        for j in speaker_folders:  # j = pxxx
            if j == 'p315':
                print("Text files for p315 do not exist, continue")
                continue
            if tune_for_specific_user:
                if j != specific_user:
                    continue

            parser.add_argument(f"--VCTK_{j}", type=str, default=f'{path_ds}{i}/{j}')
    parser.add_argument("--chars_to_ignore", type=List[str],
                        default=[
                            # ",", "?", ".", "–", "!", "-", ";", ":", "\"",   # whisper can predict punctuation, leave these out ?
                            "\\\\", "/", "“", "%", "‘", "”", "…", "...",
                            "�", "<", ">", "♪",
                            "`", "_", "^", "@", "=", "{", "~", "­", " ", "​", "¨", "£", "©", "«", "°", "´", "»",
                            "•", "™",
                            "̄", "̌", "̧", "—", "’", "„", " ", "″", "€", "℃", "⅓", "#", "$", "&", "\'", "(",
                            "■", ")", "*", "+", "[", "]"])
    parser.add_argument("--non_lang_letters", type=List[str],
                        default=["à", "q", "w", "x", "y", "ä", "é", "ì", "õ", "ö", "ō", "ŗ", "б", "в", "д", "и", "й",
                                 "к", "м", "н", "т",
                                 "ц", "ч", "ш", "щ", "ы", "я", "ё", "ख", "य", "ल", "व", "ा", "ि", "्"])
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    # TODO num_workers
    args = parser.parse_args()

    dataset = DatasetDict()
    for i in train_test_val_folders:  # i = test/train/val
        if i == 'test' or i == 'train' or i == 'val':
            speaker_folders = os.listdir(f'{path_ds}{i}')
        else:
            print(f"{i} Not train/test/val")
            continue
        for j in speaker_folders:  # j = pxxx
            if j == 'p315':
                print("Text files for p315 do not exist, continue")
                continue
            if tune_for_specific_user:
                if j != specific_user:
                    continue
            # args_dir = 'VCTK_' + j
            #TODO ?

            dataset_x = load_dataset(path=args.dataset_name, data_dir=f'{path_ds}{i}/{j}')
            # Add text_cer column to x dataset
            text_cer_x = [0.0] * len(dataset_x["train"])
            dataset_x["train"] = dataset_x["train"].add_column("text_cer", text_cer_x)
            dataset_x = dataset_x.rename_column("transcription", "sentence")
            if tune_for_specific_user:
                train_idx = [i for i in range(dataset_x["train"].shape[0]) if i % 5 != 0]
                test_idx = [i for i in range(dataset_x["train"].shape[0]) if i % 5 == 0]
                dataset["train"] = dataset_x["train"].select(train_idx)
                dataset["test"] = dataset_x["train"].select(test_idx)
            else:
                test_train_fold = i
                if i == 'val':
                    if speaker_folders.index(j) % 2 == 0:
                        test_train_fold = 'train'
                    else:
                        test_train_fold = 'test'
                if test_train_fold == 'train':
                    if not dataset.get("train"):
                        dataset["train"] = dataset_x["train"]
                    else:
                        dataset["train"] = concatenate_datasets([dataset["train"], dataset_x["train"]])
                elif test_train_fold == 'test':
                    if not dataset.get("test"):
                        dataset["test"] = dataset_x["train"]
                    else:
                        dataset["test"] = concatenate_datasets([dataset["test"], dataset_x["train"]])
                else:
                    print("huh?")

            # dataset["test"] = pp_valid["train"]

    # pp_valid = load_dataset(args.dataset_name, data_dir=args.VCTK_p225)
    # common_voice = load_dataset(args.dataset_name, data_dir=args.VCTK_p226)

    # # Add text_cer column to pp dataset
    # text_cer_pp = [0.0] * len(pp_valid["train"])
    # pp_valid["train"] = pp_valid["train"].add_column("text_cer", text_cer_pp)
    # text_cer_cv = [0.0] * len(common_voice["train"])
    # common_voice["train"] = common_voice["train"].add_column("text_cer", text_cer_cv)

    # common_voice = common_voice.rename_column("transcription", "sentence")
    # pp_valid = pp_valid.rename_column("transcription", "sentence")

    ## concatanate more datasets if you use more
    # dataset["train"] = concatenate_datasets([common_voice["train"], ...])
    # dataset["train"] = common_voice["train"]
    # dataset["test"] = pp_valid["train"]

    chars_to_ignore_regex = r''.join(args.chars_to_ignore)
    remove = str.maketrans('', '', chars_to_ignore_regex)
    text_column_name = "sentence"


    def remove_special_characters(batch, chars_to_ignore_regex, text_column_name, remove):
        if chars_to_ignore_regex is not None:
            batch[text_column_name] = batch[text_column_name].translate(remove).lower()
        else:
            batch[text_column_name] = batch[text_column_name]
        return batch


    def is_text(inputString):
        if not inputString:
            return False
        else:
            return True


    # def no_numbers(inputString):
    # 	return not any(char.isdigit() for char in inputString)

    def no_unk_letters(inputString, non_lang_letters):
        return not any(letter in inputString for letter in non_lang_letters)


    def text_cer_thresh(inputValue):
        return float(inputValue) <= 0.2


    # print("is_text")
    dataset = dataset.filter(
        is_text,
        num_proc=args.preprocessing_num_workers,
        input_columns=[text_column_name],
    )
    # print("text_cer_threshold")
    # filter rows with text_cer over threshold
    dataset = dataset.filter(
        text_cer_thresh,
        num_proc=args.preprocessing_num_workers,
        input_columns=["text_cer"],
    )
    # print("chars_to_ignore_regex ", chars_to_ignore_regex)
    # print("remove_special_characters")
    dataset = dataset.map(
        remove_special_characters,
        desc="remove special characters from datasets",
        num_proc=args.preprocessing_num_workers,
        fn_kwargs={"chars_to_ignore_regex": chars_to_ignore_regex, "text_column_name": text_column_name,
                   "remove": remove}
    )

    # print("no_unk_letters")
    # filter rows with non-target-language letters
    dataset = dataset.filter(
        no_unk_letters,
        num_proc=args.preprocessing_num_workers,
        input_columns=[text_column_name],
        fn_kwargs={"non_lang_letters": args.non_lang_letters}
    )


    def is_audio_in_length_range(length):
        return length >= 0.1 and length < 30.0


    # print("is_audio_in_length_range")
    # filter data that is shorter than min_input_length
    dataset = dataset.filter(
        is_audio_in_length_range,
        num_proc=args.preprocessing_num_workers,
        input_columns=["file_len"],
    )

    # filter rows with numbers in transcript
    # dataset = dataset.filter(
    # 	no_numbers,
    # 	num_proc=args.preprocessing_num_workers,
    # 	input_columns=[text_column_name],
    # )

    # print("feature_extr, tokenizer, processor, model")
    model_path = f"./pretrained_models/whisper-{args.model_size}"
    # model_path = f"./pretrained_models/whisper-base_en-finetuned-VCTK/17_05/checkpoint-12700"
    # model_path = f"./"
    if not os.path.exists(model_path):
        print("Bruh Wrong model path")
        exit()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, local_files_only=True, language="English",
                                                 task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True, language="English",
                                                 task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    # if torch.cuda.is_available():
    #     model = FSDP(model)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []


    def prepare_dataset(batch, feature_extractor, tokenizer):
        # print("preparing dataset")
        # print(batch['speaker_file'])
        audio = batch["audio"]
        # print("audio secured")

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # print("input features done")
        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        # print("tokenization done")

        return batch


    print("dataset")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    # print("dataset 1")
    dataset = dataset.shuffle(seed=69)
    # print("dataset 2")
    print(feature_extractor)
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"],
                          num_proc=args.preprocessing_num_workers,
                          load_from_cache_file=False,
                          fn_kwargs={'feature_extractor': feature_extractor, 'tokenizer': tokenizer})
    # dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=args.preprocessing_num_workers)
    # cache_file_names = {"train": "./data/cached-train-whisper.arrow", "test": "./data/cached-test-whisper.arrow"}
    # dataset = dataset.map(
    # 	prepare_dataset,
    # 	batched=True,
    # 	desc="prepare dataset",
    # 	remove_columns=next(iter(dataset.values())).column_names,
    # 	cache_file_names=cache_file_names
    # )

    print("data_collector, wer, cer")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")


    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}


    print("training args")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        warmup_steps=0,
        max_steps=500000,
        # sharded_ddp="simple",  # other options: zero_dp_2, zero_dp_3
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to=["tensorboard"],
        greater_is_better=False
    )

    # print("trainer")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    # print("processor save pretrained")
    processor.save_pretrained(training_args.output_dir)
    # trainer.train(resume_from_checkpoint=model_path)
    trainer.train()
