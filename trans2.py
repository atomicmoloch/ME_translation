from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric
import numpy as np

model_name = "t5-small"

ME_data = load_dataset("csv", data_files="ME_cleaned.csv")


ME_data = ME_data["train"].train_test_split(test_size=0.15)




tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    source_lang = "Middle English"
    target_lang = "English"
    prefix = "translate Middle English to modern English: "
    print(type(examples))
    inputs = []
    targets = []
    for example in examples:
        if not ((example[source_lang] is None) or (example[target_lang] is None)):
            inputs.append(prefix + example[source_lang])
            targets.append(example[target_lang])
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    print(type(model_inputs))
    return model_inputs

tokenized_text = {}
tokenized_text["train"] = preprocess_function(ME_data["train"])
tokenized_text["test"] = preprocess_function(ME_data["test"])
print(type(tokenized_text))
#tokenized_text = ME_data.map(preprocess_function)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_text["train"],
    eval_dataset=tokenized_text["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
