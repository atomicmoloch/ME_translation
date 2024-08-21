from datasets import load_dataset, load_metric
import torch

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")

print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

print(f"Memory allocated: {torch.cuda.memory_allocated(cuda_id)}")

print(f"Memory info: {torch.cuda.mem_get_info()}")

print(f"Memory reserved: {torch.cuda.memory_reserved()}")

print(f"SMI utilization: {torch.cuda.utilization()}")

source_lang = "Middle English"
target_lang = "English"
ME_data = load_dataset("csv", data_files="ME_cleaned.csv")
ME_data["train"] = ME_data["train"].filter(lambda example: (not ((example[source_lang] is None) or (example[target_lang] is None)))) #filters out rows without a full pairing
ME_data = ME_data["train"].train_test_split(test_size=0.1, seed=42) #0.1, 42


for i in range(10):
    print(ME_data["test"][i])
