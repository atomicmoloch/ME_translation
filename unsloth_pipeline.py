from datasets import load_dataset, load_metric
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import bleu

def unsloth_pipeline(model_name, evalsteps):
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

    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
            token = "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",
            )


    model = FastLanguageModel.get_peft_model(
        model,
        r = 8, #rank of finetuning process. 8 to 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head",],
        lora_alpha = 8, #equal or double  r, scaling factor for finetuning
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 42, #seed
        use_rslora = False,
        loftq_config = None,
        )

    EOS_token = tokenizer.eos_token

    def formatting_prompts_func(example):
        if isinstance(example["Middle English"], str):
            return [f"### Translate Middle English to modern English: {(example['Middle English'])} {response_template} {(example['English'])}" + EOS_token] #sometimes the trainer can pass a dict with a single entry for some reason resulting in problems like this
        output_texts = []
        for i in range(len(example['Middle English'])):
            text = f"### Translate Middle English to modern English: {(example['Middle English'][i])} {response_template} {(example['English'][i])}" + EOS_token
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
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
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
