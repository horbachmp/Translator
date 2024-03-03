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
from torch.utils.data import TensorDataset, DataLoader

#################################################################

#parameters

NUM_EPOCHS = 30
hidden_size = 256
batch_size = 64
MIN_FREQENCY = 20

###################################################################


MAX_LEN = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN_ENG = 2
EOS_TOKEN_ENG = 3

# сделано на основе туториала
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


def dataset_iterator(texts):
    # also normalize data
    for text in texts:
        text = text.lower().strip()
        text = re.sub(r"([.!?])", r" \1", text)
        text = re.sub(r"[^a-zA-Z!?]+", r" ", text)
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

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_TOKEN_ENG)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LEN):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = nn.functional.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.long().view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(dataloader, encoder, decoder, vocab_out):
    with torch.no_grad():
        decoded_words_init = []
        decoded_words_received = []
        for data in tqdm(dataloader):
            input_tensor, target_tensor = data
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()
            for decoded_sent in target_tensor:
                decoded_words = []
                for idx in decoded_sent[1:]:
                    if idx.item() == EOS_TOKEN_ENG:
                        break
                    decoded_words.append(vocab_out.lookup_token(idx.item()))
                decoded_words_init.append(' '.join(decoded_words))
            for decoded_sent in decoded_ids:
                decoded_words = []
                for idx in decoded_sent[1:]:
                    if idx.item() == EOS_TOKEN_ENG:
                        break
                    decoded_words.append(vocab_out.lookup_token(idx.item()))
                decoded_words_received.append(' '.join(decoded_words))
    print("Val example:")
    print("target:", decoded_words_init[0])
    print("received:", decoded_words_received[0])
    bleu = sacrebleu.corpus_bleu(decoded_words_received, [decoded_words_init]).score
    return bleu

def predict(dataloader, encoder, decoder, vocab_out):
    decoded_words_received = []
    with torch.no_grad():
        for input_tensor in tqdm(dataloader):
            input_tensor = input_tensor[0].to(device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()
            for decoded_sent in decoded_ids:
                decoded_words = []
                for idx in decoded_sent[1:]:
                    if idx.item() == EOS_TOKEN_ENG:
                        break
                    decoded_words.append(vocab_out.lookup_token(idx.item()))
                decoded_words_received.append(' '.join(decoded_words))
    return decoded_words_received


def train(train_dataloader, val_dataloader, test_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, n_epochs, vocab_out, print_every=5):
    print_loss_total = 0

    for epoch in range(1, n_epochs + 1):
        print(epoch)
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print("loss train:", print_loss_avg)
            bleu = validate(val_dataloader, encoder, decoder, vocab_out)
            print("val bleu:", bleu)
        
        print("Predicting...")
        translates = predict(test_loader, encoder, decoder, vocab_out)
        translates = remove_consecutive_duplicates(translates)
        file_name = "answer" + str(epoch) +".txt"
        with open(file_name, 'w') as answer_file:
            for line in translates:
                answer_file.write(line + "\n")    
        print("Predictions saved")
        torch.save(encoder.state_dict(), 'weights/encoder_' + str(epoch) +'.pt')
        torch.save(decoder.state_dict(), 'weights/decoder_' + str(epoch) +'.pt')

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

    tokenized_train_de = create_tokenized_data(train_de, vocab_de)
    tokenized_train_en = create_tokenized_data(train_en, vocab_en)
    tokenized_val_de = create_tokenized_data(val_de, vocab_de)
    tokenized_val_en = create_tokenized_data(val_en, vocab_en)
    tokenized_test_de = create_tokenized_data(test_de, vocab_de)

    train_dataset = TensorDataset(tokenized_train_de, tokenized_train_en)
    val_dataset = TensorDataset(tokenized_val_de, tokenized_val_en)
    test_dataset = TensorDataset(tokenized_test_de)

    

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print("Prepared")

    encoder = EncoderRNN(len(vocab_de), hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, len(vocab_en)).to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    print("Training...")
    train(train_loader, val_loader, test_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, NUM_EPOCHS, vocab_en, print_every=1, plot_every=10)
    print("Trained")

    print("Predicting...")
    translates = predict(test_loader, encoder, decoder, vocab_en)
    translates = remove_consecutive_duplicates(translates)
    with open('final_answer.txt', 'w') as answer_file:
        for line in translates:
            answer_file.write(line + "\n")    
    print("Predictions saved")

main()