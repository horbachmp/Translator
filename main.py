import sys
import warnings

# Отключение всех предупреждений
warnings.filterwarnings("ignore")
############################################
import matplotlib.pyplot as plt
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
import heapq
#################################################################

#parameters

NUM_EPOCHS = 20
MIN_FREQENCY = 2

torch.manual_seed(0)

SRC_VOCAB_SIZE = 0
TGT_VOCAB_SIZE = 0

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 1024
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


###################################################################


MAX_LEN = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN_ENG = 2
EOS_TOKEN_ENG = 3
PAD_TOKEN_ENG = 0
PAD_TOKEN_DE = 0

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

class NoiseDataset(TensorDataset):
    def __init__(self, src_texts, tgt_texts, vocab_src, noise_level=0.1, p_noise=0.1, crop_level=0.2, p_crop=0.1):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.noise_level = noise_level
        self.p_noise = p_noise
        self.p_crop = p_crop
        self.crop_level = crop_level
        self.tokens = list(vocab_src.get_stoi().values())
        super().__init__()

    def add_noise_to_text(self, text_tensor):
        # print(text_tensor)
        num_noise_chars = math.ceil(text_tensor.shape[0] * self.noise_level)
        for _ in range(num_noise_chars):
            index = random.randint(0, text_tensor.shape[0] - 1)
            # print(text_tensor.shape)
            text_tensor[index] = self.tokens[random.randint(0,len(self.tokens) - 1)]
        return text_tensor

    def add_random_crop(self, text_tensor):
        num_noise_chars = math.ceil(text_tensor.shape[0] * self.crop_level)
        for _ in range(num_noise_chars):
            index = random.randint(0, text_tensor.shape[0] - 1)
            text_tensor = torch.cat((text_tensor[:index], text_tensor[index+1:], torch.Tensor([PAD_TOKEN_DE])), dim=0)
        return text_tensor


    # def change_to_unk(self, text_tensor):


    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        if random.random() < self.p_noise:
            src_text = self.add_noise_to_text(src_text)
        if src_text.shape[0] > 30 and random.random() < self.p_crop:
            src_text = self.add_random_crop(src_text)

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
    losses = []
    bleus = []
    for epoch in range(1, n_epochs + 1):
        print(epoch)
        loss = train_epoch(train_dataloader, model, optimizer, criterion)
        losses.append(loss)
        print_loss_total += loss
        scheduler.step()
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print("loss train:", print_loss_avg)
        bleu = validate(val_dataloader, val_answers, model, vocab_out)
        print("val bleu:", bleu)
        bleus.append(bleu)

        print(predict_one("vielen dank .", model, vocab_in, vocab_out))
        print("Predicting...")
        translates = predict(test_loader, model, vocab_out)
        translates = remove_consecutive_duplicates(translates)
        file_name = "answer" + str(epoch) +".txt"
        with open(file_name, 'w', encoding="utf8") as answer_file:
            for line in translates:
                answer_file.write(line + "\n")
        print("Predictions saved")
        torch.save(model.state_dict(), 'weights/model_' + str(epoch) +'.pt')
    
      # График для лосса
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 1), losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('train_loss_plot_'+ str(epoch) + '.png')
        # plt.show()

        # График для BLEU Score
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch + 1), bleus, label='Val BLEU Score')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU Score Value')
        plt.title('Validation BLEU Score')
        plt.legend()
        plt.grid(True)
        plt.savefig('val_bleu_plot_'+ str(epoch) + '.png')
        # plt.show()

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


def beam_predict_one(input_tensor, model, beam_width):
    input_tensor = input_tensor.to(device)
    src = input_tensor
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)
    src_mask_pad = (src == PAD_TOKEN_ENG).type(torch.bool).to(device)
    memory = model.encode(src, src_mask, src_mask_pad)

    ys = torch.ones(src.shape[0], 1).fill_(SOS_TOKEN_ENG).type(torch.long).to(device)
    options = []
    heapq.heapify(options)

    memory = memory.to(device)
    tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                .type(torch.bool)).to(device)
    out = model.decode(ys, memory, tgt_mask)
    prob = torch.log(nn.functional.softmax(model.generator(out[:, -1]))[0])
    probs, next_words = torch.topk(prob, k=beam_width, dim=0)
    for i in range(beam_width):
        heapq.heappush(options, (probs[i], torch.cat([ys, next_words[i].unsqueeze(0).unsqueeze(0)], dim=1).tolist()))
    final_options = []
    heapq.heapify(final_options)
    for _ in range(MAX_LEN):
        if len(options) == 0:
          break
        while len(options) > beam_width:
            heapq.heappop(options)
        while len(final_options) > beam_width:
            heapq.heappop(final_options)
        new_options = []
        heapq.heapify(new_options)
        for elem in options:
            best_prob, best_seq = elem
            best_seq = torch.tensor(best_seq, device=device)
            tgt_mask = (generate_square_subsequent_mask(best_seq.size(1)).type(torch.bool)).to(device)
            out = model.decode(best_seq, memory, tgt_mask)
            prob = torch.log(nn.functional.softmax(model.generator(out[:, -1]))[0])
            probs, next_words = torch.topk(prob, k=beam_width, dim=0)
            for i in range(beam_width):
                try:
                    if next_words[i].item() == EOS_TOKEN_ENG:
                        heapq.heappush(final_options, (probs[i] + best_prob, torch.cat([best_seq, next_words[i].unsqueeze(0).unsqueeze(0)], dim=1).tolist()))
                    else:
                        heapq.heappush(new_options, (probs[i] + best_prob, torch.cat([best_seq, next_words[i].unsqueeze(0).unsqueeze(0)], dim=1).tolist()))
                except:
                    pass
                if len(options) > beam_width:
                    heapq.heappop(options)
                if len(final_options) > beam_width:
                    heapq.heappop(final_options)
        options = new_options
    while len(options) > 0:
        max_el = heapq.nlargest(1, options)[0]
        options.remove(max_el)
        heapq.heappush(final_options, max_el)
    max_el = heapq.nlargest(1, final_options)[0]
    best_prob, best_seq = max_el
    return best_seq[0]

def beam_validate(dataloader, val_answers, model, vocab_out, beam_width=2):
    model.eval()
    with torch.no_grad():
        decoded_words_received = []
        for data in tqdm(dataloader):
            input_tensor, target_tensor = data
            lengths = torch.count_nonzero(input_tensor, dim=1)
            for batch_id in range(input_tensor.shape[0]):
                curr_input_tensor = input_tensor[batch_id].unsqueeze(0).to(device)
                ans = beam_predict_one(curr_input_tensor, model, beam_width)
                decoded_words = []
                for idx in ans[1:]:
                    if idx == EOS_TOKEN_ENG:
                        break
                    decoded_words.append(vocab_out.lookup_token(idx))
                decoded_words_received.append(' '.join(decoded_words[:lengths[batch_id]+5]))
    print("Val example:")
    i = random.randint(0, BATCH_SIZE - 3)
    print("target:", val_answers[i][:-1])
    print("received:", decoded_words_received[i])
    bleu = sacrebleu.corpus_bleu(decoded_words_received, [val_answers]).score
    return bleu

def beam_predict_one_sent(sent, model, vocab_in, vocab_out, beam_width=3):
    sent_tokens = create_tokenized_data([sent], vocab_in)
    model.eval()
    decoded_words_received = []
    with torch.no_grad():
        input_tensor = sent_tokens.clone()
        input_tensor = input_tensor.to(device)
        lengths = torch.count_nonzero(input_tensor, dim=1)
        ans = beam_predict_one(input_tensor, model, beam_width)
        decoded_words = []
        for idx in ans[1:]:
            if idx == EOS_TOKEN_ENG:
                break
            decoded_words.append(vocab_out.lookup_token(idx))
        decoded_words_received.append(' '.join(decoded_words[:lengths[0] + 5]))
    return decoded_words_received


def beam_predict(dataloader, model, vocab_out, beam_width=2):
    model.eval()
    with torch.no_grad():
        decoded_words_received = []
        for data in tqdm(dataloader):
            input_tensor = data[0]
            lengths = torch.count_nonzero(input_tensor, dim=1)
            for batch_id in range(input_tensor.shape[0]):
                curr_input_tensor = input_tensor[batch_id].unsqueeze(0).to(device)
                ans = beam_predict_one(curr_input_tensor, model, beam_width)
                decoded_words = []
                for idx in ans[1:]:
                    if idx == EOS_TOKEN_ENG:
                        break
                    decoded_words.append(vocab_out.lookup_token(idx))
                decoded_words_received.append(' '.join(decoded_words[:lengths[batch_id]+5]))
    return decoded_words_received


def main():
    torch.cuda.empty_cache()

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
    train_de = open('data/data/train.de-en.de', encoding="utf8").readlines()
    train_en = open('data/data/train.de-en.en', encoding="utf8").readlines()
    val_de = open('data/data/val.de-en.de', encoding="utf8").readlines()
    val_en = open('data/data/val.de-en.en', encoding="utf8").readlines()
    test_de = open('data/data/test1.de-en.de', encoding="utf8").readlines()

    ## adding train to val
    # train_de.extend(val_de)
    # train_en.extend(val_en)


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
    PAD_TOKEN_DE = vocab_de['<pad>']
    tokenized_train_de = create_tokenized_data(train_de, vocab_de)
    tokenized_train_en = create_tokenized_data(train_en, vocab_en)
    tokenized_val_de = create_tokenized_data(val_de, vocab_de)
    tokenized_val_en = create_tokenized_data(val_en, vocab_en)
    tokenized_test_de = create_tokenized_data(test_de, vocab_de)

    train_dataset = NoiseDataset(tokenized_train_de, tokenized_train_en, vocab_de)
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

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0004, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
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
    with open('final_answer.txt', 'w', encoding="utf8") as answer_file:
        for line in translates:
            answer_file.write(line + "\n")
    print("Predictions saved")

main()