import torch
from torch import nn
from utils import utils, datautils
import config


class GlobalRes:
    def __init__(self, type_vocab_file, word_vecs_file):
        self.type_vocab, self.type_id_dict = datautils.load_type_vocab(type_vocab_file)
        self.parent_type_ids_dict = utils.get_parent_type_ids_dict(self.type_id_dict)
        self.n_types = len(self.type_vocab)

        print('loading {} ...'.format(word_vecs_file), end=' ', flush=True)
        self.token_vocab, self.token_vecs = datautils.load_pickle_data(word_vecs_file)
        self.token_id_dict = {t: i for i, t in enumerate(self.token_vocab)}
        print('done', flush=True)
        self.zero_pad_token_id = self.token_id_dict[config.TOKEN_ZERO_PAD]
        self.mention_token_id = self.token_id_dict[config.TOKEN_MENTION]
        self.unknown_token_id = self.token_id_dict[config.TOKEN_UNK]
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(self.token_vecs))
        self.embedding_layer.padding_idx = self.zero_pad_token_id
        self.embedding_layer.weight.requires_grad = False
        self.embedding_layer.share_memory()


def anchor_samples_to_model_samples(samples, mention_token_id, parent_type_ids_dict):
    model_samples = list()
    for i, sample in enumerate(samples):
        mstr = sample[1]
        full_labels = utils.get_full_type_ids(sample[5], parent_type_ids_dict)
        model_samples.append(get_model_sample(
            mention_id=sample[0], mention_str=mstr, mention_span=[sample[2], sample[3]], sent_tokens=sample[6],
            mention_token_id=mention_token_id, labels=full_labels))
    return model_samples
