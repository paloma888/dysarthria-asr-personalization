from pipeline import processor
import torch
import string
import jiwer

def inference(model, audio, device):
    input_features = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features.to(device)
    with torch.no_grad():
      tokens = model.generate(input_features, language="en", task="transcribe", max_new_tokens=64)
    
    predicted_text = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return predicted_text


def scoring_normalize(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation)).strip()
    text = " ".join(text.split())
    return text

def get_metrics(model, test_set):
    actual_list_iso = []
    pred_list_iso = []
    actual_list_cont = []
    pred_list_cont = []

    for sample in test_set:
        prediction = inference(model, sample["audio"], "cuda")
        actual = sample["text"]

        prediction = scoring_normalize(prediction)
        actual = scoring_normalize(actual)

        if sample["is_isolated"] == True:
            actual_list_iso.append(actual)
            pred_list_iso.append(prediction)
        else:
            actual_list_cont.append(actual)
            pred_list_cont.append(prediction)

    wer_cont = jiwer.wer(actual_list_cont, pred_list_cont)
    cer_cont = jiwer.cer(actual_list_cont, pred_list_cont)
    cer_iso = jiwer.cer(actual_list_iso, pred_list_iso)

    matches = sum(1 for p, q in zip(pred_list_iso, actual_list_iso) if p == q)
    wra_iso = matches / len(pred_list_iso)

    return wer_cont, cer_cont, wra_iso, cer_iso


if __name__ == "__main__":
    print(scoring_normalize("Yet he still thinks, as swiftly as ever."))
    print(scoring_normalize("He's still"))
    print(scoring_normalize("UP"))