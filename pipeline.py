#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
#from datasets import load_dataset, load_metric
#import numpy as np
#import bleu
# from trl import SFTConfig, SFTTrainer,  DataCollatorForCompletionOnlyLM
#from unsloth import FastLanguageModel
#import torch
from seq2seq_pipeline import *
#from unsloth_pipeline import *
#from gguf_pipeline import *
from causualLM_pipeline import *



def pipeline_4eval(model_name):
    seq2seq_pipeline(model_name, 799)

def pipeline_1eval(model_name):
    seq2seq_pipeline(model_name, 3196)


def causualLM_pipeline_4eval(model_name):
    unsloth_pipeline(model_name, 799)

def causualLM_pipeline_1eval(model_name):
    unsloth_pipeline(model_name, 3196)

def causualLM_pipeline_nounsloth(model_name):
    cllm_pipeline(model_name, 3196)

def gguf_pipeline_1eval(model_name, filename):
    gguf_pipeline(model_name, filename, 3196)
