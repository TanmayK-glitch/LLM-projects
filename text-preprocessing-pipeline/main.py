import re
import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken

# 1. Read raw text
with open("book.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2. Creating Custom Tokenizer
class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]

        ids = [self.str_to_int[token] for token in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

print("--- Testing Custom Tokenizer ---")
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

tokenizer = Tokenizer(vocab)
encoded = tokenizer.encode(raw_text)
print("Encoded sample (first 50 tokens):", encoded[:50])
print("Decoded sample:", tokenizer.decode(encoded[:50]))


# 3. PyTorch Dataset and DataLoader with tiktoken
print("\n--- Testing PyTorch Dataset & DataLoader with tiktoken ---")
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.output_ids = []

        # tokenizer here is expected to be an encoder with an `encode` method
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            output_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.output_ids.append(torch.tensor(output_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # tiktoken uses the name 'gpt2' for the GPT-2 encoding
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Inputs:\n", inputs)
print("Outputs:\n", targets)

# 4. Token Embedding layer
print("\n--- Token Embedding Layer ---")
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print("Inputs shape:", inputs.shape)
print("Embedding layer weights shape:", token_embedding_layer.weight.shape)
print("\nScript execution completed successfully.")
