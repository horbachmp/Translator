import sys
import warnings
import torch
import torch.nn as nn
import gdown
import zipfile
import os
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from typing import Type
from torch.optim.lr_scheduler import StepLR, OneCycleLR

from torch.utils.data import TensorDataset, DataLoader
import random



from data_process import *
from config import *
from model import *
from beam_search import *
from utils import *
from train import *








def main():
    torch.manual_seed(0)
    random.seed(42)
    np.random.seed(42)
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
    UNK_TOKEN_DE = vocab_de['<unk>']
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

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    max_lr = 0.0004  # Максимальная скорость обучения
    total_steps = len(train_loader) * NUM_EPOCHS  # Общее количество шагов обучения
    pct_start = 0.3  # Процент шагов для увеличения скорости обучения
    anneal_strategy = 'cos'  # Стратегия изменения скорости обучения (cosine)

    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start, anneal_strategy=anneal_strategy)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ENG)
    if not os.path.exists("weights/"):
        os.makedirs("weights/")


    print("Training...")
    best_epoch = train(train_loader, val_loader, val_en, test_loader, transformer, optimizer, scheduler, criterion, NUM_EPOCHS, vocab_en, vocab_de, print_every=1)
    print("Trained")
    if BEAM_SEARCH:
        print("Best epoch:", best_epoch)
        print("Predicting with beam search...")
        weights_path = 'weights/model_' + str(best_epoch) +'.pt'
        transformer.load_state_dict(torch.load(weights_path))
        transformer = transformer.to(device)

        preds = beam_predict(test_loader, transformer, vocab_en, beam_width=2)
        file_name = "beam_answer_" + str(best_epoch) +".txt"
        with open(file_name, 'w', encoding="utf8") as answer_file:
            for line in preds:
                answer_file.write(line + "\n")
        print("Predictions saved")

main()