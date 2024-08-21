from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric
import numpy as np

ME_data = load_dataset("csv", data_files="train-0.csv")

datasets_train_test = ME_data["train"].train_test_split(test_size=3000)
datasets_train_validation = datasets_train_test["train"].train_test_split(test_size=3000)

ME_data["train"] = datasets_train_validation["train"]
ME_data["validation"] = datasets_train_validation["test"]
ME_data["test"] = datasets_train_test["test"]

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#infer = AutoModelForSeq2SeqLM.from_pretrained("Qilex/bart-large-me-en")

batch_size = 8
model_dir = f"/home/moloch/Documents/transformer/tb-{model_name}"

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps", #Evaluation is done and logged each eval_steps
    eval_steps=100,
    logging_strategy="steps", #Logging done each every logging_steps
    logging_steps=100,
    save_strategy="steps", #Save is done one every save_steps. 'epoch' would save at each epoch
    save_steps=200, #Number of updates steps before 2 checkpoint saves
    learning_rate=4e-5, #Initial learning rate for AdamW optimizer - weight decay implementation
    per_device_train_batch_size=batch_size, #batch size per core/CPU for training
    per_device_eval_batch_size=batch_size, #batch size per core/CPU for evaluation
    weight_decay=0.01, #weight decay to apply at all layers. penalizes overfitting complexity
    save_total_limit=3,
    num_train_epochs=1, #number of training epochs to perform. an epoch is the number of times algo will work through the entire training dataset
    predict_with_generate=True,
    load_best_model_at_end=True, #loads best found model at end of training
    metric_for_best_model="rouge1", #metric to compare 2 models. greater-is-better defualts to true
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)
metric = load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}



def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


trainer = Seq2SeqTrainer(
    model_init=model_init, #fun that instantiates model to be used. each call to train will start a new instance of the model as given
    args=args, #trainingarguments. will default to basic instance of trainingarguments with output_dir set to tmp_trainer if not provided
    train_dataset=ME_data["train"],
    eval_dataset=ME_data["validation"], #dataset. columns not accepted by model.forward() are removed
    data_collator=data_collator, #function to form a batch from the train_dataset and eval_dataset elements
    tokenizer=tokenizer, #tokenizer used to preprocess data. if provided, will automatically pad inputs to max length when batching inputs
    compute_metrics=compute_metrics #function used to compute metrics at evaluation. takes an evalprediction and returns a dict string to metric values
)

# Start TensorBoard before training to monitor it in progress



trainer.train()


def translate(sentence):
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    outputs = infer.generate(input_ids, max_new_tokens = len(sentence.split(' '))*10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

