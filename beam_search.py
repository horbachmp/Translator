from tqdm import tqdm
import heapq 
import torch.nn as nn
import sacrebleu
import random

from data_process import create_tokenized_data
from utils import generate_square_subsequent_mask
from config import *

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
