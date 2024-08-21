from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric
#import nltk


model_dir = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

text = input("Enter text to translate: ")
inputs = ["translate Middle English to modern English: " + text]

inputs = tokenizer(inputs, max_length=128, truncation=True, return_tensors="pt")

print(type(inputs))

output = model.generate(**inputs, num_beams=8, do_sample=True)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
#modern_text = nltk.sent_tokenize(decoded_output.strip())[0]

print(decoded_output)
print(type(decoded_output))
