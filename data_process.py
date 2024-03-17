import torch
from torch.utils.data import TensorDataset, DataLoader
import random
import math


from config import *


def dataset_iterator(texts):
    for text in texts:
        yield text.strip().split()

def create_tokenized_data(texts, vocab):
    train_tokens = []
    for text in dataset_iterator(texts):
        tokens = [vocab['<sos>']] + [vocab[word] if word in vocab else vocab['<unk>'] for word in text]
        tokens.append(vocab['<eos>'])
        train_tokens += [tokens]
    tokenized = torch.full((len(train_tokens), MAX_LEN), vocab['<pad>'], dtype=torch.int32)
    for i, tokens in enumerate(train_tokens):
        tokenized[i, :min(MAX_LEN, len(tokens))] = torch.tensor(tokens[:min(MAX_LEN, len(tokens))])
    return tokenized


class NoiseDataset(TensorDataset):
    def __init__(self, src_texts, tgt_texts, vocab_src, noise_level=0.1, p_noise=0.1, crop_level=0.2, p_crop=0.1, unk_level=0.1, p_unk=0.1):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.noise_level = noise_level
        self.p_noise = p_noise
        self.p_crop = p_crop
        self.crop_level = crop_level
        self.unk_level = unk_level
        self.p_unk = p_unk
        self.tokens = list(vocab_src.get_stoi().values())
        super().__init__()

    def add_noise_to_text(self, text_tensor, text_len):
        # print(text_tensor)
        num_noise_chars = math.ceil(text_len * self.noise_level)
        for _ in range(num_noise_chars):
            index = random.randint(0, text_len - 1)
            # print(text_tensor.shape)
            text_tensor[index] = self.tokens[random.randint(0,len(self.tokens) - 1)]
        return text_tensor

    def add_random_crop(self, text_tensor, text_len):
        num_noise_chars = math.ceil(text_len * self.crop_level)
        for _ in range(num_noise_chars):
            index = random.randint(0, text_len - 1)
            text_tensor = torch.cat((text_tensor[:index], text_tensor[index+1:], torch.Tensor([PAD_TOKEN_DE])), dim=0)
        return text_tensor


    def change_to_unk(self, text_tensor, text_len):
        num_noise_chars = math.ceil(text_len * self.unk_level)
        for _ in range(num_noise_chars):
            index = random.randint(0, text_len - 1)
            text_tensor[index] = UNK_TOKEN_DE
        return text_tensor


    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        text_len = torch.count_nonzero(src_text)
        if NOISE and text_len > 10 and random.random() < self.p_noise:
            src_text = self.add_noise_to_text(src_text, text_len)
        if CROP and text_len > 10 and random.random() < self.p_crop:
            src_text = self.add_random_crop(src_text, text_len)
        if ADD_UNK and text_len > 10 and random.random() < self.p_unk:
            src_text = self.change_to_unk(src_text, text_len)

        return src_text, tgt_text

    def __len__(self):
        return len(self.src_texts)
    

