import sys
import warnings

# Отключение всех предупреждений
warnings.filterwarnings("ignore")
############################################

import torch
import torch.nn as nn
import gdown
import zipfile
import os
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from typing import Type
from torch.optim.lr_scheduler import StepLR
# import matplotlib.pyplot as plt
from tqdm import tqdm
import sacrebleu
import re
import math
from torch.utils.data import TensorDataset, DataLoader
import random
#################################################################

#parameters

NUM_EPOCHS = 30
MIN_FREQENCY = 20

torch.manual_seed(0)

SRC_VOCAB_SIZE = 0
TGT_VOCAB_SIZE = 0

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

###################################################################


MAX_LEN = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN_ENG = 2
EOS_TOKEN_ENG = 3
PAD_TOKEN_ENG = 0

# сделано на основе туториала
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


def dataset_iterator(texts):
    # also normalize data
    for text in texts:
        text = text.lower().strip()
        # text = re.sub(r"([.!?])", r" \1", text)
        # text = re.sub(r"[^a-zA-Z!?]+", r" ", text)
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
        # print(token_embedding.shape)
        # print(self.pos_embedding.shape)
        # print(self.pos_embedding[:token_embedding.size(1), :].transpose(0,1).shape)
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(1), :].transpose(0,1))

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
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

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    # src_padding_mask = (src == PAD_TOKEN_ENG).transpose(0, 1)
    # tgt_padding_mask = (tgt == PAD_TOKEN_ENG).transpose(0, 1)
    src_padding_mask = (src == PAD_TOKEN_ENG)
    tgt_padding_mask = (tgt == PAD_TOKEN_ENG)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def train_epoch(dataloader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        tgt_input = target_tensor[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(input_tensor, tgt_input)

        optimizer.zero_grad()

        logits = model(src=input_tensor, trg=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,src_padding_mask = src_padding_mask, tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

#         print(torch.max(logits[0], dim = 1)[1])

        tgt_out = target_tensor[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1).long())
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(dataloader, val_answers, model, vocab_out):
    model.eval()
    with torch.no_grad():
        decoded_words_received = []
        for data in tqdm(dataloader):
            input_tensor, target_tensor = data
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            lengths = torch.count_nonzero(input_tensor, dim=1)
#             print("lengths", lengths.shape)
            src = input_tensor
            num_tokens = src.shape[1]
            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)
            src_mask_pad = (src == PAD_TOKEN_ENG).type(torch.bool).to(device)
            memory = model.encode(src, src_mask, src_mask_pad)

            ys = torch.ones(src.shape[0], 1).fill_(SOS_TOKEN_ENG).type(torch.long).to(device)
            for i in range(MAX_LEN):
                memory = memory.to(device)
                tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                            .type(torch.bool)).to(device)
                out = model.decode(ys, memory, tgt_mask)
                prob = model.generator(out[:, -1])
#                 print(prob.shape)
#                 print(prob[0])
                _, next_words = torch.max(prob, dim=1)
#                 print(next_words.shape)
                next_words = next_words.unsqueeze(0).transpose(0,1)
                ys = torch.cat([ys, next_words], dim=1)
#             print(ys)
#             print(EOS_TOKEN_ENG)
#             print(torch.sum(ys==EOS_TOKEN_ENG))
            i = 0
            for decoded_sent in ys:
                decoded_words = []
                for idx in decoded_sent[1:]:
                    if idx.item() == EOS_TOKEN_ENG:
#                         print("hi")
                        break
                    decoded_words.append(vocab_out.lookup_token(idx.item()))
                decoded_words_received.append(' '.join(decoded_words[:lengths[i]+5]))
                i += 1
    print("Val example:")
    i = random.randint(0, BATCH_SIZE - 3)
    print("target:", val_answers[i][:-1])
    print("received:", decoded_words_received[i])
    bleu = sacrebleu.corpus_bleu(decoded_words_received, [val_answers]).score
    return bleu

def predict(dataloader, model, vocab_out):
    model.eval()
    decoded_words_received = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_tensor = data[0]
            input_tensor = input_tensor.to(device)
            lengths = torch.count_nonzero(input_tensor, dim=1)
            src = input_tensor
            num_tokens = src.shape[1]
            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)
            src_mask_pad = (src == PAD_TOKEN_ENG).type(torch.bool).to(device)

            memory = model.encode(src, src_mask, src_mask_pad)
            ys = torch.ones(src.shape[0], 1).fill_(SOS_TOKEN_ENG).type(torch.long).to(device)
            for i in range(MAX_LEN):
                memory = memory.to(device)
                tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                            .type(torch.bool)).to(device)
                out = model.decode(ys, memory, tgt_mask)
                prob = model.generator(out[:, -1])
                _, next_words = torch.max(prob, dim=1)
                next_words = next_words.unsqueeze(0).transpose(0,1)
                ys = torch.cat([ys, next_words], dim=1)
            i = 0
            for decoded_sent in ys:
                decoded_words = []
                for idx in decoded_sent[1:]:
                    if idx.item() == EOS_TOKEN_ENG:
                        break
                    decoded_words.append(vocab_out.lookup_token(idx.item()))
                decoded_words_received.append(' '.join(decoded_words[:lengths[i]+5]))
                i+= 1
    return decoded_words_received


def train(train_dataloader, val_dataloader, val_answers, test_loader, model, optimizer, scheduler, criterion, n_epochs, vocab_out, vocab_in, print_every=5):
    print_loss_total = 0

    for epoch in range(1, n_epochs + 1):
        print(epoch)
        loss = train_epoch(train_dataloader, model, optimizer, criterion)
        print_loss_total += loss
        scheduler.step()
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print("loss train:", print_loss_avg)
            bleu = validate(val_dataloader, val_answers, model, vocab_out)
            print("val bleu:", bleu)

        print(predict_one("vielen dank .", model, vocab_in, vocab_out))
        print("Predicting...")
        translates = predict(test_loader, model, vocab_out)
        translates = remove_consecutive_duplicates(translates)
        file_name = "answer" + str(epoch) +".txt"
        with open(file_name, 'w') as answer_file:
            for line in translates:
                answer_file.write(line + "\n")
        print("Predictions saved")
        torch.save(model.state_dict(), 'weights/model_' + str(epoch) +'.pt')

def remove_consecutive_duplicates(lines):
    filtered_lines = []
    for line in lines:
        words = line.split()
        filtered_words = []
        for i, word in enumerate(words):
            if i < 1 or word != words[i - 1]:
                filtered_words.append(word)
        filtered_line = ' '.join(filtered_words)
        filtered_lines.append(filtered_line)
    return filtered_lines


def predict_one(sent, model, vocab_in, vocab_out):
    sent_tokens = create_tokenized_data([sent], vocab_in)
    model.eval()
    decoded_words_received = []
    with torch.no_grad():
        input_tensor = sent_tokens.clone()
        input_tensor = input_tensor.to(device)
        lengths = torch.count_nonzero(input_tensor, dim=1)
        src = input_tensor
        num_tokens = src.shape[1]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)
        src_mask_pad = (src == PAD_TOKEN_ENG).type(torch.bool).to(device)

        memory = model.encode(src, src_mask, src_mask_pad)
        ys = torch.ones(src.shape[0], 1).fill_(SOS_TOKEN_ENG).type(torch.long).to(device)
        for i in range(MAX_LEN):
            memory = memory.to(device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                        .type(torch.bool)).to(device)
            out = model.decode(ys, memory, tgt_mask)
            prob = model.generator(out[:, -1])
            _, next_words = torch.max(prob, dim=1)
            next_words = next_words.unsqueeze(0).transpose(0,1)
            ys = torch.cat([ys, next_words], dim=1)
        i = 0
        for decoded_sent in ys:
            decoded_words = []
            for idx in decoded_sent[1:]:
                if idx.item() == EOS_TOKEN_ENG:
                    break
                decoded_words.append(vocab_out.lookup_token(idx.item()))
            decoded_words_received.append(' '.join(decoded_words[:lengths[i] + 5]))
            i += 1
    return decoded_words_received


def main():
    print("Device:", device)

    print("Downloading data...")
    if not os.path.exists('data.zip'):
        url = 'https://drive.google.com/uc?id=1_TGzGyCNcozHYUXPzniR9D4GGZbD43X2'
        output = 'data.zip'
        gdown.download(url, output, quiet=False)
    zip_file_path = 'data.zip'
    extract_to_folder = 'data'
    os.makedirs(extract_to_folder, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
    train_de = open('data/data/train.de-en.de').readlines()
    train_en = open('data/data/train.de-en.en').readlines()
    val_de = open('data/data/val.de-en.de').readlines()
    val_en = open('data/data/val.de-en.en').readlines()
    test_de = open('data/data/test1.de-en.de').readlines()
    print("Downloaded")
    print("Preparing data...")
    vocab_en = build_vocab_from_iterator(
        dataset_iterator(train_en),
        specials=['<pad>', '<unk>', '<sos>', '<eos>'], min_freq=MIN_FREQENCY,
    )
    vocab_de = build_vocab_from_iterator(
        dataset_iterator(train_de),
        specials=['<pad>', '<unk>', '<sos>', '<eos>'], min_freq=MIN_FREQENCY,
    )
    SOS_TOKEN_ENG = vocab_en['<sos>']
    EOS_TOKEN_ENG = vocab_en['<eos>']
    PAD_TOKEN_ENG = vocab_en['<pad>']
    tokenized_train_de = create_tokenized_data(train_de, vocab_de)
    tokenized_train_en = create_tokenized_data(train_en, vocab_en)
    tokenized_val_de = create_tokenized_data(val_de, vocab_de)
    tokenized_val_en = create_tokenized_data(val_en, vocab_en)
    tokenized_test_de = create_tokenized_data(test_de, vocab_de)

    train_dataset = TensorDataset(tokenized_train_de, tokenized_train_en)
    val_dataset = TensorDataset(tokenized_val_de, tokenized_val_en)
    test_dataset = TensorDataset(tokenized_test_de)



    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print("Prepared")

    SRC_VOCAB_SIZE = len(vocab_de)
    TGT_VOCAB_SIZE = len(vocab_en)

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0004, betas=(0.9, 0.98), eps=1e-9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ENG)
    if not os.path.exists("weights/"):
        os.makedirs("weights/")


    print("Training...")
    train(train_loader, val_loader, val_en, test_loader, transformer, optimizer, scheduler, criterion, NUM_EPOCHS, vocab_en, vocab_de, print_every=1)
    print("Trained")

    print("Predicting...")
    translates = predict(test_loader, transformer, vocab_en)
    translates = remove_consecutive_duplicates(translates)
    with open('final_answer.txt', 'w') as answer_file:
        for line in translates:
            answer_file.write(line + "\n")
    print("Predictions saved")

main()