from platform import platform
from os.path import join
import socket

if platform().startswith('Windows'):
    PLATFORM = 'Windows'
    RES_DIR = 'd:/data/res'
    DATA_DIR = 'd:/data/fet'
    EL_DATA_DIR = 'd:/data/el'
else:
    PLATFORM = 'Linux'
    RES_DIR = '/data/hldai/res'
    DATA_DIR = '/home/hldai/data/fet'
    EL_DATA_DIR = '/home/hldai/data/el'

TOKEN_UNK = '<UNK>'
TOKEN_ZERO_PAD = '<ZPAD>'
TOKEN_EMPTY_PAD = '<EPAD>'
TOKEN_MENTION = '<MEN>'

RANDOM_SEED = 771
NP_RANDOM_SEED = 7711
PY_RANDOM_SEED = 9973

MACHINE_NAME = socket.gethostname()
MODEL_DIR = join(DATA_DIR, 'models')
RESULT_DIR = join(DATA_DIR, 'results')

EL_CANDIDATES_DATA_FILE = join(RES_DIR, 'enwiki-20151002-candidate-gen.pkl')
WIKI_FETEL_WORDVEC_FILE = join(RES_DIR, 'enwiki-20151002-nef-wv-glv840B300d.pkl')

FIGER_FILES = {
    'anchor-train-data-prefix': join(DATA_DIR, 'Wiki/enwiki20151002anchor-fetwiki-0_1'),
}

BBN_FILES = {
}
