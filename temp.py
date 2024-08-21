from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset, load_metric
import numpy as np
import bleu


bleu_metric = load_metric("bleu")
source_lang = "Middle English"
target_lang = "English"
ME_data = load_dataset("csv", data_files="ME_cleaned.csv")
ME_data["train"] = ME_data["train"].filter(lambda example: (not ((example[source_lang] is None) or (example[target_lang] is None)))) #filters out rows without a full pairing
ME_data = ME_data["train"].train_test_split(test_size=0.1, seed=42) #0.1, 42
print(ME_data)
