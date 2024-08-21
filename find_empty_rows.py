from datasets import load_dataset

ME_data = load_dataset("csv", data_files="train-0.csv")

source_lang = "Middle English"
target_lang = "English"

for example in ME_data["train"]:
    if (example[source_lang] is None):
        print(example[target_lang])
    if (example[target_lang] is None):
        print(example[source_lang])
