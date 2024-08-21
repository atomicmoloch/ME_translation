from datasets import load_dataset, load_metric
from peft import PeftModel, PeftConfig
from trl import SFTTrainer
#from unsloth import is_bfloat16_supported
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
import bleu


# def causualLM_pipeline(model_name, evalsteps):
#     metric = load_metric("bleu")
#     source_lang = "Middle English"
#     target_lang = "English"
#     ME_data = load_dataset("csv", data_files="ME_cleaned.csv")
#     ME_data["train"] = ME_data["train"].filter(lambda example: (not ((example[source_lang] is None) or (example[target_lang] is None)))) #filters out rows without a full pairing
#
#     ME_data = ME_data["train"].train_test_split(test_size=0.999, seed=42)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#
#     def preprocess_function(examples):
#         prefix = "translate Middle English to modern English: "
#         inputs = []
#         targets = []
#         for example in examples[source_lang]:
#             inputs.append(prefix + example)
#         for example in examples[target_lang]:
#             targets.append(example)
#         model_inputs = tokenizer(inputs, max_length=128, truncation=True)
#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(targets, max_length=128, truncation=True)
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs
#
#     def compute_metrics(eval_pred): #computes bleu score
#         predictions, labels = eval_pred
#         predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
#         decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#
#         labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#         print(decoded_labels[0] + " : " + decoded_predictions[0])
#
#         split_predictions = [prediction.split(' ') for prediction in decoded_predictions]
#         split_labels = [[label.split(' ')] for label in decoded_labels]
#
#         results = metric.compute(predictions=split_predictions, references=split_labels)
#         return results
#
#     tokenized_text = ME_data.map(preprocess_function, batched=True)
#     tokenized_text["test"] = tokenized_text["test"].remove_columns(["Middle English", "English"])
#     tokenized_text["train"] = tokenized_text["train"].remove_columns(["Middle English", "English"])
#     print(tokenized_text["train"][0]['labels'])
#
#
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#
#
#     training_args = TrainingArguments(
#         output_dir="./" + model_name + "-results",
#         evaluation_strategy="steps",
#         eval_steps=evalsteps,
#         save_strategy="steps",
#         save_steps=799,
#         learning_rate=2e-5,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         weight_decay=0.01,
#         save_total_limit=3,
#         num_train_epochs=1,
#         metric_for_best_model="bleu",
#    #     predict_with_generate=True,
#     #generation_max_length=128, #if more generation settings are needed, instead create and pass a GenerationConfig object
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         compute_metrics=compute_metrics,
#         train_dataset=tokenized_text["train"],
#         eval_dataset=tokenized_text["test"],
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#     )
#
#     trainer.train()





# def causualLM_pipeline_PEFT(model_name, evalsteps):
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     torch.cuda.empty_cache()
#     print(device)
#
#
#     metric = load_metric("bleu")
#     source_lang = "Middle English"
#     target_lang = "English"
#     ME_data = load_dataset("csv", data_files="ME_cleaned.csv")
#     ME_data["train"] = ME_data["train"].filter(lambda example: (not ((example[source_lang] is None) or (example[target_lang] is None)))) #filters out rows without a full pairing
#
#     ME_data = ME_data["train"].train_test_split(test_size=0.1, seed=42)
#
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#
#     response_template = "###  Translation: "
#
#     def formatting_prompts_func(example):
#         output_texts = []
#         for i in range(len(example['Middle English'])):
#             text = f"### Translate Middle English to modern English: {example['Middle English'][i]} {response_template} {example['English'][i]}"
#             output_texts.append(text)
#        # print(output_texts)
#         return output_texts
#     #instruction_template = "### Translate Middle English to modern English:"
#
#
#
# #    tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
#     initial_token_count = len(tokenizer)
#     added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
#     model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)
#
#
#     collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
#
#     arguments = SFTConfig(
#         output_dir="./" + model_name + "-results",)
#
#     trainer = SFTTrainer(
#         model,
#         tokenizer=tokenizer,
#         train_dataset=ME_data["train"],
#         args=arguments,
#         formatting_func=formatting_prompts_func,
#         data_collator=collator,
#     )
#
#     trainer.train()
#
#
  # torch.cuda.empty_cache()
  #   max_seq_length = 128


    # metric = load_metric("bleu")
    # source_lang = "Middle English"
    # target_lang = "English"
    # ME_data = load_dataset("csv", data_files="ME_cleaned.csv")
    # ME_data["train"] = ME_data["train"].filter(lambda example: (not ((example[source_lang] is None) or (example[target_lang] is None))))
    # response_template = "###  Translation: "
    #
    #
    # ME_data = ME_data["train"].train_test_split(test_size=0.1, seed=42)
    #
    #
    # def compute_metrics(eval_pred): #computes bleu score
    #     predictions, labels = eval_pred
    #
    #     predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    #     decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    #
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #
    #     print(decoded_labels[0] + " : " + decoded_predictions[0])
    #
    #     split_predictions = [prediction.split(' ') for prediction in decoded_predictions]
    #     split_labels = [[label.split(' ')] for label in decoded_labels]
    #
    #     results = metric.compute(predictions=split_predictions, references=split_labels)
    #     return results




#    tokenized_text = ME_data.map(formatting_prompts_func, batched=True)
#    tokenized_text["test"] = tokenized_text["test"].remove_columns(["Middle English", "English"])
#    tokenized_text["train"] = tokenized_text["train"].remove_columns(["Middle English", "English"])
#    print(tokenized_text["train"][0]['labels'])
def cllm_pipeline(model_name, evalsteps):
    torch.cuda.empty_cache()
    max_seq_length = 128


    metric = load_metric("bleu")
    source_lang = "Middle English"
    target_lang = "English"
    ME_data = load_dataset("csv", data_files="ME_cleaned.csv")
    ME_data["train"] = ME_data["train"].filter(lambda example: (not ((example[source_lang] is None) or (example[target_lang] is None))))
    response_template = "###  Translation: "


    ME_data = ME_data["train"].train_test_split(test_size=0.1, seed=42)


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




#    tokenized_text = ME_data.map(formatting_prompts_func, batched=True)
#    tokenized_text["test"] = tokenized_text["test"].remove_columns(["Middle English", "English"])
#    tokenized_text["train"] = tokenized_text["train"].remove_columns(["Middle English", "English"])
#    print(tokenized_text["train"][0]['labels'])

    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
 #   config = PeftConfig.from_pretrained(model_name)

 #  model = PeftModel.from_pretrained(model, config)

    EOS_token = tokenizer.eos_token
    print(EOS_token)
    tokenizer.pad_token = tokenizer.eos_token #added for gpt. does not like custom pad tokens
   # tokenizer.add_special_tokens({'pad_token': '<|   pad   |>'}) #gpt demanded this
    print(tokenizer.pad_token)

    def formatting_prompts_func(example):
        if isinstance(example["Middle English"], str):
            return [f"### Translate Middle English to modern English: {(example['Middle English'])} {response_template} {(example['English'])}" + "<EOS>" + EOS_token] #sometimes the trainer can pass a dict with a single entry for some reason resulting in problems like this
        output_texts = []
        for i in range(len(example['Middle English'])):
            text = f"### Translate Middle English to modern English: {(example['Middle English'][i])} {response_template} {(example['English'][i])}" + "<EOS>" + EOS_token #changed for GPT, change back later
            output_texts.append(text)
        return output_texts
       # print(output_texts)
  #  EOS_token = tokenizer.eos_token


  #  tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
  #  model.config.pad_token_id = tokenizer.pad_token_id # updating model config
  #  tokenizer.padding_side = 'right' # padding to right (otherwise SFTTrainer shows warning)

    arguments = TrainingArguments(
        output_dir="./" + model_name + "-results",
        weight_decay=0.01,
        seed=42,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        learning_rate = 2e-4,
        fp16 = True,
        bf16 = False,
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
        metric_for_best_model="bleu",
     #   predict_with_generate=True, #not implemented yet by huggingface
        )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        args = arguments,
      #  dataset_text_field = "labels",
        max_seq_length = max_seq_length,
        train_dataset=ME_data["train"],
        formatting_func=formatting_prompts_func,
        compute_metrics=compute_metrics,
        )
    torch.cuda.empty_cache()
    trainer.train()
    # trainer.train(resume_from_checkpoint = True) #change back later
