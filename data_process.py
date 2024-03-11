import torch
import torch.nn as nn
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
    

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(1), :].transpose(0,1))

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask, src_mask_pad):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask, src_mask_pad)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)