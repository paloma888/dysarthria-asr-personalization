import re
import random
from pathlib import Path
import soundfile as sf
from datasets import Dataset, Audio
from collections import defaultdict

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def prompt_from_txt(path) -> str | None:
    with open(path) as f: text = f.read()
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return None
    text = re.sub(r"\[.*?\]", "", text)
    text = text.strip()
    return text

def wav_prompt_pair(folder_path) -> list[dict]:
    pairs = []
    for wavpath in sorted(Path(folder_path).glob("*.wav")):
        promptpath = wavpath.parent.parent / "prompts" / (wavpath.stem + ".txt")
        if not promptpath.exists():
            continue
        text = prompt_from_txt(promptpath)
        if text is not None:
            pairs.append({"audio": str(wavpath), "text": text})

    return pairs


def gather_torgo(torgoroot) -> list[dict]:
    final_pairs = []
    for group in Path(torgoroot).iterdir():
        if not group.is_dir():
            continue

        for person in group.iterdir():
            if not person.is_dir():
                continue

            for session in person.iterdir():
                if not session.is_dir() or session.name == "Notes":
                    continue
                
                wav_folder_path = session / "wav_arrayMic"
                if not wav_folder_path.exists():
                    continue

                pairs = wav_prompt_pair(wav_folder_path)

                #add speaker and isolated vs. continuous to each pair
                for pair in pairs:
                    pair["person"] = person.name
                    pair["is_isolated"] = len(pair["text"].split()) == 1
                final_pairs.extend(pairs)
    
    
    return final_pairs

def group_by_speaker(pairs: list[dict]) -> defaultdict:
    groups = defaultdict(list)
    for p in pairs:
        groups[p["person"]].append(p)

    return groups

def train_val_test_split(speaker_groups: defaultdict):
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

if __name__ == "__main__":
    # print(prompt_from_txt('../data/torgo/F/F01/Session1/prompts/0007.txt'))
    # pairs = wav_prompt_pair("../data/TORGO/F/F01/Session1/wav_arrayMic")

    pairs = gather_torgo('../data/torgo')
    print(len(pairs))
    groups = group_by_speaker(pairs)
    print({k: len(v) for k, v in groups.items()})

    train, val, test = train_val_test_split(groups)
    print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"total: {len(train) + len(val) + len(test)}")
