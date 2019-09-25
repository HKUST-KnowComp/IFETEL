import torch
from torch import nn
# from collections import namedtuple
import numpy as np
from typing import List
import random
from utils import utils, datautils
import config


class ModelSample:
    def __init__(self, mention_id, mention_str, mstr_token_seq, context_token_seq, mention_token_idx):
        self.mention_id = mention_id
        self.mention_str = mention_str
        self.mstr_token_seq = mstr_token_seq
        self.context_token_seq = context_token_seq
        self.mention_token_idx = mention_token_idx


class LabeledModelSample(ModelSample):
    def __init__(self, mention_id, mention_str, mstr_token_seq, context_token_seq, mention_token_idx, labels):
        super().__init__(mention_id, mention_str, mstr_token_seq, context_token_seq, mention_token_idx)
        self.labels = labels

# ModelSample = namedtuple('ModelSample', [
#     'mention_id', 'mention_str', 'mstr_token_seq', 'context_token_seq', 'mention_token_idx', 'labels'])


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
        # self.embedding_layer.share_memory()


def get_model_sample(mention_id, mention_str, mention_span, sent_tokens, mention_token_id):
    pos_beg, pos_end = mention_span
    mstr_tokens = sent_tokens[pos_beg:pos_end]
    mention_token_idx = pos_beg
    context_token_seq = sent_tokens[:pos_beg] + [mention_token_id] + sent_tokens[pos_end:]

    if len(context_token_seq) > 256:
        context_token_seq = context_token_seq[:256]
    if mention_token_idx >= 256:
        mention_token_idx = 255

    return ModelSample(mention_id, mention_str, mstr_tokens, context_token_seq, mention_token_idx)


def get_labeled_model_sample(mention_id, mention_str, mention_span, sent_tokens, mention_token_id, labels):
    s = get_model_sample(mention_id, mention_str, mention_span, sent_tokens, mention_token_id)
    return LabeledModelSample(s.mention_id, s.mention_str, s.mstr_token_seq, s.context_token_seq, s.mention_token_idx,
                              labels)


def anchor_samples_to_model_samples(samples, mention_token_id, parent_type_ids_dict):
    model_samples = list()
    for i, sample in enumerate(samples):
        mstr = sample[1]
        full_labels = utils.get_full_type_ids(sample[5], parent_type_ids_dict)
        model_samples.append(get_labeled_model_sample(
            mention_id=sample[0], mention_str=mstr, mention_span=[sample[2], sample[3]], sent_tokens=sample[6],
            mention_token_id=mention_token_id, labels=full_labels))
    return model_samples


def model_samples_from_json(token_id_dict, unknown_token_id, mention_token_id, type_id_dict,
                            mentions_file, sents_file):
    sent_tokens_dict = datautils.read_sents_to_token_id_seq_dict(
        sents_file, token_id_dict, unknown_token_id)

    samples = list()
    mentions = datautils.read_json_objs(mentions_file)
    for m in mentions:
        sample = get_model_sample(m['mention_id'], mention_str=m['str'], mention_span=m['span'],
                                  sent_tokens=sent_tokens_dict[m['sent_id']], mention_token_id=mention_token_id)
        samples.append(sample)
    return samples


def get_mstr_cxt_batch_input(samples: List[ModelSample]):
    context_token_seqs = [s.context_token_seq for s in samples]
    mention_token_idxs = [s.mention_token_idx for s in samples]
    mstrs = [s.mention_str for s in samples]
    mstr_token_seqs = [s.mstr_token_seq for s in samples]
    return context_token_seqs, mention_token_idxs, mstrs, mstr_token_seqs


def get_mstr_cxt_label_batch_input(device, n_types, samples: List[LabeledModelSample]):
    tmp = get_mstr_cxt_batch_input(samples)
    y_true = torch.tensor([utils.onehot_encode(s.labels, n_types) for s in samples],
                          dtype=torch.float32, device=device)
    return (*tmp, y_true)


def get_person_type_loss_vec(l2_person_type_ids, n_types, per_penalty, device):
    person_loss_vec = np.ones(n_types, np.float32)
    for tid in l2_person_type_ids:
        person_loss_vec[tid] = per_penalty
    return torch.tensor(person_loss_vec, dtype=torch.float32, device=device)


def get_mstr_context_batch_input_rand_per(device, n_types, samples: List[LabeledModelSample], person_type_id,
                                          person_l2_type_ids):
    context_token_seqs = [s.context_token_seq for s in samples]
    mention_token_idxs = [s.mention_token_idx for s in samples]
    mstrs = [s.mention_str for s in samples]
    mstr_token_seqs = [s.mstr_token_seq for s in samples]
    type_vecs = list()
    for sample in samples:
        type_vec = utils.onehot_encode(sample.labels, n_types)
        if person_type_id is not None and person_type_id in sample.labels:
            for _ in range(3):
                rand_person_type_id = person_l2_type_ids[random.randint(0, len(person_l2_type_ids) - 1)]
                if type_vec[rand_person_type_id] < 1.0:
                    type_vec[rand_person_type_id] = 1.0
                    break
        type_vecs.append(type_vec)
    type_vecs = torch.tensor(type_vecs, dtype=torch.float32, device=device)
    return context_token_seqs, mention_token_idxs, mstrs, mstr_token_seqs, type_vecs
