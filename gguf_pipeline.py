from datasets import load_dataset, load_metric
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer,  DataCollatorForCompletionOnlyLM
import torch
import bleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling



def gguf_pipeline1(model_name, filename, evalsteps):
    metric = load_metric("bleu")
    source_lang = "Middle English"
    target_lang = "English"
    ME_data = load_dataset("csv", data_files="ME_cleaned.csv")
    ME_data["train"] = ME_data["train"].filter(lambda example: (not ((example[source_lang] is None) or (example[target_lang] is None)))) #filters out rows without a full pairing

    ME_data = ME_data["train"].train_test_split(test_size=0.999, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(model_name, gguf_file=filename)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def preprocess_function(examples):
        prefix = "translate Middle English to modern English: "
        inputs = []
        targets = []
        for example in examples[source_lang]:
            inputs.append(prefix + example)
        for example in examples[target_lang]:
            targets.append(example)
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(eval_pred): #computes bleu score
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print(decoded_labels[0] + " : " + decoded_predictions[0])

        split_predictions = [prediction.split(' ') for prediction in decoded_predictions]
        split_labels = [[label.split(' ')] for label in decoded_labels]

        results = metric.compute(predictions=split_predictions, references=split_labels)
        return results

    tokenized_text = ME_data.map(preprocess_function, batched=True)
    tokenized_text["test"] = tokenized_text["test"].remove_columns(["Middle English", "English"])
    tokenized_text["train"] = tokenized_text["train"].remove_columns(["Middle English", "English"])
    print(tokenized_text["train"][0]['labels'])


    model = AutoModelForCausalLM.from_pretrained(model_name, gguf_file=filename)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    training_args = TrainingArguments(
        output_dir="./" + model_name + "-results",
        evaluation_strategy="steps",
        eval_steps=evalsteps,
        save_strategy="steps",
        save_steps=799,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        metric_for_best_model="bleu",
   #     predict_with_generate=True,
    #generation_max_length=128, #if more generation settings are needed, instead create and pass a GenerationConfig object
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_text["train"],
        eval_dataset=tokenized_text["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


def gguf_pipeline(model_name, filename, evalsteps):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    print(device)


    metric = load_metric("bleu")
    source_lang = "Middle English"
    target_lang = "English"
    ME_data = load_dataset("csv", data_files="ME_cleaned.csv")
    ME_data["train"] = ME_data["train"].filter(lambda example: (not ((example[source_lang] is None) or (example[target_lang] is None)))) #filters out rows without a full pairing

    ME_data = ME_data["train"].train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name, gguf_file=filename)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(model_name, gguf_file=filename)

    response_template = "###  Translation: "

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['Middle English'])):
            text = f"### Translate Middle English to modern English: {example['Middle English'][i]} {response_template} {example['English'][i]}"
            output_texts.append(text)
       # print(output_texts)
        return output_texts
    #instruction_template = "### Translate Middle English to modern English:"



#    tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
    initial_token_count = len(tokenizer)
    added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
    model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)


    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    arguments = SFTConfig(
        output_dir="./" + model_name + "-results",)

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=ME_data["train"],
        args=arguments,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()
