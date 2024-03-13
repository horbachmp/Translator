import torch

NUM_EPOCHS = 20
MIN_FREQENCY = 2

SRC_VOCAB_SIZE = 0
TGT_VOCAB_SIZE = 0

EMB_SIZE = 1024
NHEAD = 8
FFN_HID_DIM = 1024
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

BEAM_SEARCH = True

NOISE = True
CROP = True
ADD_UNK = True

#####################################################

MAX_LEN = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN_ENG = 2
EOS_TOKEN_ENG = 3
PAD_TOKEN_ENG = 0
PAD_TOKEN_DE = 0
UNK_TOKEN_DE = 1
