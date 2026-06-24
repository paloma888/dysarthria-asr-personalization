import re
from pathlib import Path

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


if __name__ == "__main__":
    # print(prompt_from_txt('../data/torgo/F/F01/Session1/prompts/0007.txt'))
    pairs = wav_prompt_pair("../data/TORGO/F/F01/Session1/wav_arrayMic")
    print(f"Found {len(pairs)} pairs")
    for p in pairs[:5]:
        print(p)
