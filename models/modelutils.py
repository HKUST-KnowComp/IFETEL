import torch
from torch import nn
import numpy as np


def get_seqs_torch_input(device, seqs):
    seq_lens = torch.tensor([len(seq) for seq in seqs], dtype=torch.long, device=device)
    id_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in seqs]
    id_seqs = torch.nn.utils.rnn.pad_sequence(id_seqs, batch_first=True)
    return id_seqs, seq_lens


def get_len_sorted_context_seqs_input(device, context_seqs, mention_token_idxs):
    data_tups = list(enumerate(zip(context_seqs, mention_token_idxs)))
    data_tups.sort(key=lambda x: -len(x[1][0]))
    seqs = [x[1][0] for x in data_tups]
    mention_token_idxs = [x[1][1] for x in data_tups]
    idxs = [x[0] for x in data_tups]
    back_idxs = [0] * len(idxs)
    for i, idx in enumerate(idxs):
        back_idxs[idx] = i

    back_idxs = torch.tensor(back_idxs, dtype=torch.long, device=device)
    seqs, seq_lens = get_seqs_torch_input(device, seqs)
    mention_token_idxs = torch.tensor(mention_token_idxs, dtype=torch.long, device=device)
    return seqs, seq_lens, mention_token_idxs, back_idxs


def get_avg_token_vecs(device, embedding_layer: nn.Embedding, token_seqs):
    lens = torch.tensor([len(seq) for seq in token_seqs], dtype=torch.float32, device=device
                        ).view(-1, 1)
    seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in token_seqs]
    seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                           padding_value=embedding_layer.padding_idx)
    token_vecs = embedding_layer(seqs)
    vecs_avg = torch.div(torch.sum(token_vecs, dim=1), lens)
    return vecs_avg


def init_lstm_hidden(device, batch_size, hidden_dim, bidirectional):
    d = 2 if bidirectional else 1
    return (torch.zeros(d, batch_size, hidden_dim, requires_grad=True, device=device),
            torch.zeros(d, batch_size, hidden_dim, requires_grad=True, device=device))


def build_hierarchy_vecs(type_vocab, type_to_id_dict):
    from utils import utils

    n_types = len(type_vocab)
    l1_type_vec = np.zeros(n_types, np.float32)
    l1_type_indices = list()
    child_type_vecs = np.zeros((n_types, n_types), np.float32)
    for i, t in enumerate(type_vocab):
        p = utils.get_parent_type(t)
        if p is None:
            l1_type_indices.append(i)
            l1_type_vec[type_to_id_dict[t]] = 1
        else:
            child_type_vecs[type_to_id_dict[p]][type_to_id_dict[t]] = 1
    l1_type_indices = np.array(l1_type_indices, np.int32)
    return l1_type_indices, l1_type_vec, child_type_vecs
