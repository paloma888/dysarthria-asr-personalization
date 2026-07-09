import re
import random
import string
from pathlib import Path
import soundfile as sf
from datasets import Dataset, Audio
from collections import defaultdict

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

from dataclasses import dataclass
from typing import Any

#keeping punctuation now for natural predictions, will need to strip punctuation at eval time
def normalize_text(text: str) -> str:
    return text.lower().strip()


def prompt_from_txt_torgo(path) -> str | None:
    with open(path) as f: text = f.read()
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return None
    text = re.sub(r"\[.*?\]", "", text)
    text = text.strip()
    return text

def wav_prompt_pair_torgo(folder_path) -> list[dict]:
    pairs = []
    for wavpath in sorted(Path(folder_path).glob("*.wav")):
        promptpath = wavpath.parent.parent / "prompts" / (wavpath.stem + ".txt")
        if not promptpath.exists():
            continue
        text = prompt_from_txt_torgo(promptpath)
        if text is not None:
            text = normalize_text(text)
            pairs.append({"audio": str(wavpath), "text": text})

    return pairs

def filter_valid_audio(pairs: list[dict]) -> list[dict]:
    final_pairs = []
    bad_files = 0
    for item in pairs:
        try:
            sf.info(item["audio"])
            final_pairs.append(item)
        except Exception as e:
            bad_files += 1
    
    print(f"Dropped {bad_files} bad files.")
    return final_pairs


def gather_torgo(torgoroot) -> list[dict]:
    final_pairs = []
    for group in sorted(Path(torgoroot).iterdir()):
        if not group.is_dir():
            continue

        for person in sorted(group.iterdir()):
            if not person.is_dir():
                continue

            for session in sorted(person.iterdir()):
                if not session.is_dir() or session.name == "Notes":
                    continue
                
                wav_folder_path = session / "wav_arrayMic"
                if not wav_folder_path.exists():
                    continue

                pairs = wav_prompt_pair_torgo(wav_folder_path)

                #add speaker and isolated vs. continuous to each pair
                for pair in pairs:
                    pair["person"] = person.name
                    pair["is_isolated"] = len(pair["text"].split()) == 1
                final_pairs.extend(pairs)
    
    
    return final_pairs


def load_mlf_ua(mlf_path) -> dict:
    lines = open(mlf_path).readlines()
    stem_to_prompt = {}

    for i in range(1, len(lines) - 1):
        curr_line = lines[i].strip()
        if curr_line.startswith('"'):
            if lines[i+1].strip() == "." or lines[i+1].strip().startswith('"'):
                continue
            stem = Path(curr_line.strip('"')).stem
            stem_to_prompt[stem] = lines[i+1].strip()

    return stem_to_prompt
            

def gather_uaspeech(audioroot, mlfroot) -> list[dict]:
    final_pairs = []
    normalized_audio_path = Path(audioroot) / "normalized"

    for person in sorted(normalized_audio_path.iterdir()):
        if not person.is_dir():
            continue
        
        speaker_mlf_path = Path(mlfroot) / person.name / f"{person.name}_word.mlf"
        if not speaker_mlf_path.exists():
            continue
        stem_to_prompt = load_mlf_ua(speaker_mlf_path)

        for wavpath in sorted(person.glob("*.wav")):
            stem = wavpath.stem
            if stem.endswith("_M5"):
                text = stem_to_prompt.get(stem)
                if text:
                    text = normalize_text(text)
                    final_pairs.append({
                        "audio": str(wavpath), 
                        "text": text, 
                        "person": person.name, 
                        "is_isolated": True
                    })
    
    return final_pairs



def group_by_speaker(pairs: list[dict]) -> defaultdict:
    groups = defaultdict(list)
    for p in pairs:
        groups[p["person"]].append(p)

    return groups

def train_val_test_split_torgo(speaker_groups: defaultdict):
    random.seed(20)
    train, val, test = [], [], []
    for person, info_dicts in speaker_groups.items():
        random.shuffle(info_dicts)

        size = len(info_dicts)
        endof_train = int(size * .8)
        endof_val = int(size * .9)

        train.extend(info_dicts[:endof_train])
        val.extend(info_dicts[endof_train:endof_val])
        test.extend(info_dicts[endof_val:])
    
    return train, val, test


def train_val_test_split_ua(pairs: list[dict]):
    train, b2 = [], []
    for pair in pairs:
        block = Path(pair["audio"]).stem.split("_")[1]
        if block == "B1" or block == "B3":
            train.append(pair)
        elif block == "B2":
            b2.append(pair)
    
    random.seed(20)
    random.shuffle(b2)
    midpoint_b2 = len(b2) // 2
    val = b2[:midpoint_b2]
    test = b2[midpoint_b2:]

    return train, val, test



def dataset_from_pairs(pairs: list[dict]) -> Dataset:
    ds = Dataset.from_list(pairs)
    #resample to 16kHz automatically in case resampling needed
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

def build_training_info(row: dict) -> dict:
    audio = row["audio"]
    #spectrogram
    row["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    #token ids
    row["labels"] = processor.tokenizer(row["text"]).input_ids
    return row


@dataclass
class DataCollatorSpeechSeq2Seq:
  processor: Any
  def __call__(self, features):
    input_features = [{"input_features": f["input_features"]} for f in features]
    batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
    label_features = [{"input_ids": f["labels"]} for f in features]
    labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
      labels = labels[:, 1:]
    batch["labels"] = labels
    return batch
  

if __name__ == "__main__":
    # print(prompt_from_txt('../data/torgo/F/F01/Session1/prompts/0007.txt'))
    # pairs = wav_prompt_pair("../data/TORGO/F/F01/Session1/wav_arrayMic")

    torgo = gather_torgo('../data/torgo')
    torgo = filter_valid_audio(torgo)
    print(f"total torgo samples {len(torgo)}")

    ua = gather_uaspeech('../data/UASpeech/audio', '../data/UASpeech/mlf')
    ua = filter_valid_audio(ua)

    torgo_groups = group_by_speaker(torgo)
    train_torgo, val_torgo, test_torgo = train_val_test_split_torgo(torgo_groups)

    train_ua, val_ua, test_ua = train_val_test_split_ua(ua)
    total_train = train_torgo + train_ua
    total_val = val_torgo + val_ua
    total_test = test_torgo + test_ua
    cont_test = sum(1 for p in total_test if not p["is_isolated"])
    print("total continuous in test set: ", cont_test)
    # print(f"total UASpeech samples: {len(ua)}")
    # print("speakers:", sorted(set(p["person"] for p in ua)))
    # print("sample:", ua[0])
    # groups = group_by_speaker(pairs)
    # print({k: len(v) for k, v in groups.items()})

    # train, val, test = train_val_test_split(groups)
    # print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")
    # print(f"total: {len(train) + len(val) + len(test)}")

    # mics = set()
    # for p in ua:
    #     stem = Path(p["audio"]).stem
    #     mics.add(stem.split("_")[-1])
    # print("mics present in results:", mics)
    print(f"train: {len(total_train)}, val: {len(total_val)}, test: {len(total_test)}")
    print(f"total utterances: {len(total_train)+len(total_val)+len(total_test)}")

