from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset, load_metric
import numpy as np
import bleu
from tqdm import tqdm
#import evaluate


bleu_metric = load_metric("bleu")
rouge_metric = load_metric("rouge", seed=42)
#print(rouge_metric.inputs_description)
source_lang = "Middle English"
target_lang = "English"
ME_data = load_dataset("csv", data_files="ME_cleaned.csv")
ME_data["train"] = ME_data["train"].filter(lambda example: (not ((example[source_lang] is None) or (example[target_lang] is None)))) #filters out rows without a full pairing
ME_data = ME_data["train"].train_test_split(test_size=0.1, seed=42) #0.1, 42


def bleu_calculate(predictions, labels):
    print(predictions[0] + " : " + labels[0])
    split_predictions = [prediction.split(' ') for prediction in predictions]
    split_labels = [[label.split(' ')] for label in labels]
    results = bleu_metric.compute(predictions=split_predictions, references=split_labels)
    return results

def rouge_calculate(predictions, labels):
    results = rouge_metric.compute(predictions=predictions, references=labels, use_aggregator=True)
    return results

def write_to_file(predictions, rc, bl, model_dir):
    with open(model_dir + 'prediction_results.txt', 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")
    with open(model_dir + 'test_results.txt', 'w') as f:
        f.write(f"{rc}\n{bl}")


def seq2seq_test(model_dir, beam_num=1, do_sample=False, early_stopping=False):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    inputs = ["translate Middle English to modern English: " + sentence for sentence in (ME_data["test"])["Middle English"]]
    tokenized_inputs = [tokenizer(input_, truncation=True, padding=True, return_tensors="pt") for input_ in inputs]
    predictions = []
    for tokenized_input in tokenized_inputs:
        predictions.append( (model.generate(**tokenized_input, max_length=tokenizer.model_max_length, num_beams=beam_num, do_sample=do_sample, early_stopping=early_stopping))[0] )
        print(str(len(predictions)), end="\r")
    #tokenized_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True, return_tensors="pt")
    #predictions = model.generate(**tokenized_inputs, max_length=128, num_beams=beam_num, do_sample=do_sample, early_stopping=early_stopping)
 #   predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True) #may need to change to True again
    return (rouge_calculate(decoded_predictions, (ME_data["test"])["English"]), bleu_calculate(decoded_predictions, (ME_data["test"])["English"]))


def causualLM_test(model_dir, gpt2=False, beam_num=1, do_sample=False, early_stopping=False):
   # torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    if gpt2:
        model.to('cuda') #not needed on unsloth models
    prefix = "### Translate Middle English to modern English: "
    response_template = " ###  Translation: "
    promptlen = len(tokenizer(prefix + response_template, truncation=True, return_tensors="pt"))
    print(f"Model Max Length : {tokenizer.model_max_length}")
    tokenized_inputs = [tokenizer((prefix + sentence + response_template), truncation=True, return_tensors="pt") for sentence in (ME_data["test"])["Middle English"]] #padding=True
   # tokenized_inputs = [tokenizer((prefix + ((ME_data["test"])["Middle English"])[0] + response_template), truncation=True, padding=True, return_tensors="pt")] #this was test code

    #model.config.pad_token_id = model.config.eos_token_id #needed for gpt 2
    #tokenizer.pad_token = tokenizer.eos_token #needed for gpt 2
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.config.pad_token_id = tokenizer.pad_token_id
    EOS_token = tokenizer.eos_token


    print("Generating Predictions: ")

    tokenized_predictions = []

    for i in tqdm (range(len(tokenized_inputs))):
        tokenized_input = (tokenized_inputs[i]).to('cuda')
        # tokenized_predictions.append((model.generate(**tokenized_input, min_length=0, max_length=tokenizer.model_max_length, num_beams=beam_num, do_sample=do_sample, early_stopping=early_stopping))[0])4
        tokenized_predictions.append((model.generate(**tokenized_input, min_length=0, max_length=256+promptlen, num_beams=beam_num, do_sample=do_sample, early_stopping=early_stopping))[0])
   #     print(tokenizer.decode(tokenized_predictions[i]))



 #   tokenized_input in tokenized_inputs:
 #       tokenized_input = tokenized_input.to('cuda')
 #       tokenized_predictions.append((model.generate(**tokenized_input, min_length=0, max_length=256, num_beams=beam_num, do_sample=do_sample, early_stopping=early_stopping))[0])


   #     predictions.append((((model.generate(**tokenized_input, min_length=0, max_length=128, num_beams=beam_num, do_sample=do_sample, early_stopping=early_stopping))[0]).split(response_template))[1])

    decoded_predictions = tokenizer.batch_decode(tokenized_predictions, skip_special_tokens=True)

    decoded_predictions = [(prediction.split(response_template))[1] for prediction in decoded_predictions]
    if gpt2:
        decoded_predictions = [(prediction.split("<EOS>"))[0] for prediction in decoded_predictions] #added for GPT 2

 #   print(decoded_predictions)
    rc = rouge_calculate(decoded_predictions, (ME_data["test"])["English"])
    bl = bleu_calculate(decoded_predictions, (ME_data["test"])["English"])
    write_to_file(decoded_predictions, rc, bl, model_dir)

    return (rc, bl)


#print(seq2seq_test("facebook/bart-base"))
#print(causualLM_test("./unsloth/tinyllama-bnb-4bit-results/checkpoint-25586/"))
#print(causualLM_test("unsloth/Phi-3-mini-4k-instruct-v0-bnb-4bit"))
print(causualLM_test("./openai-community/gpt2-results/checkpoint-25586/", True))
#print("Baseline: ")
#print(rouge_calculate((ME_data["test"])["Middle English"], (ME_data["test"])["English"]), bleu_calculate((ME_data["test"])["Middle English"], (ME_data["test"])["English"]))
#print("---")
#print("mt5 greedy: ")
#print(seq2seq_test("./google/mt5-base-results/checkpoint-3196/"))
#print("---\nbart base:")
#print(seq2seq_test("./facebook/bart-base-results/checkpoint-3196/"))
#print("---\nt5 base:")
#print(seq2seq_test("./results1/checkpoint-3000/"))
