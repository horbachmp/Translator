from tqdm import tqdm
import torch
import torch.nn as nn
import sacrebleu
import random
import matplotlib.pyplot as plt
import numpy as np

from config import *
from utils import create_mask, generate_square_subsequent_mask, remove_consecutive_duplicates
from data_process import create_tokenized_data

def train_epoch(dataloader, model, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = 0
    lrs = []
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        tgt_input = target_tensor[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(input_tensor, tgt_input)

        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
            lrs.append(scheduler.get_last_lr())

        logits = model(src=input_tensor, trg=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,src_padding_mask = src_padding_mask, tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

        tgt_out = target_tensor[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1).long())
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader), lrs

def validate_bleu(dataloader, val_answers, model, vocab_out):
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

def validate_loss(dataloader, model, criterion):
    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        tgt_input = target_tensor[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(input_tensor, tgt_input)

        logits = model(src=input_tensor, trg=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,src_padding_mask = src_padding_mask, tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

        tgt_out = target_tensor[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1).long())

        total_loss += loss.item()

    return total_loss / len(dataloader)

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
    val_losses = []
    bleus = []
    lrs = []
    lrs.append(scheduler.get_last_lr())
    for epoch in range(1, n_epochs + 1):
        print(epoch)
        loss, lrs_epoch = train_epoch(train_dataloader, model, optimizer, criterion, scheduler)
        losses.append(loss)
        print_loss_total += loss
        scheduler.step()
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print("loss train:", print_loss_avg)
        bleu = validate_bleu(val_dataloader, val_answers, model, vocab_out)
        val_loss = validate_loss(val_dataloader, model, criterion)
        val_losses.append(val_loss)
        print("val bleu:", bleu)
        bleus.append(bleu)
        lrs.extend(lrs_epoch)

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
        plt.plot(range(1, epoch + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Training & Val Loss')
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
        
        # График для lr
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(lrs)), lrs, label='LR')
        plt.xlabel('step')
        plt.ylabel('LR')
        plt.title('LR')
        plt.legend()
        plt.grid(True)
        plt.savefig('lr_plot_'+ str(epoch) + '.png')
        # plt.show()

    return np.argmax(bleus) + 1




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