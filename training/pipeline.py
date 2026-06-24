import re

def prompt_from_txt(path) -> str | None:
    with open(path) as f: text = f.read()
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return None
    text = re.sub(r"\[.*?\]", "", text)
    text = text.strip()
    return text

if __name__ == "__main__":
    print(prompt_from_txt('../data/torgo/F/F01/Session1/prompts/0007.txt'))
