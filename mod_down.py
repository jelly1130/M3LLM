from transformers import AutoModel, AutoTokenizer

model_name = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"Model '{model_name}' downloaded successfully.")
