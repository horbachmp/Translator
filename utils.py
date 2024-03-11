import torch

from config import device, PAD_TOKEN_ENG

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