import torch
from torch import nn
import torch.nn.functional as F
from models import aaa, modelutils


class NEFAAA(aaa.AAA):
    def __init__(self, device, type_vocab, type_id_dict, embedding_layer: nn.Embedding, context_lstm_hidden_dim,
                 type_embed_dim, dropout=0.5, use_mlp=False, mlp_hidden_dim=None):
        super(NEFAAA, self).__init__(device, type_vocab, type_id_dict, embedding_layer, context_lstm_hidden_dim,
                                     type_embed_dim, dropout)
        self.linear_att = None
        self.use_mlp = use_mlp

        linear_map_input_dim = 2 * self.context_lstm_hidden_dim + self.word_vec_dim + self.n_types
        if not self.use_mlp:
            self.linear_map = nn.Linear(linear_map_input_dim, type_embed_dim, bias=False)
            # self.linear_map = nn.Linear(linear_map_input_dim, self.n_types, bias=False)
        else:
            mlp_hidden_dim = linear_map_input_dim // 2 if mlp_hidden_dim is None else mlp_hidden_dim
            self.linear_map1 = nn.Linear(linear_map_input_dim, mlp_hidden_dim)
            self.lin1_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map2 = nn.Linear(mlp_hidden_dim, type_embed_dim)

    def forward(self, training, context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_sgns,
                debug_flg=False):
        batch_size = len(context_token_seqs)

        context_token_seqs, seq_lens, mention_token_idxs, back_idxs = modelutils.get_len_sorted_context_seqs_input(
            self.device, context_token_seqs, mention_token_idxs)
        # self.context_hidden = self.init_context_hidden(batch_size)
        context_lstm_output = self.get_context_lstm_output(
            context_token_seqs, seq_lens, mention_token_idxs, batch_size)
        context_lstm_output = context_lstm_output[back_idxs]

        name_output = modelutils.get_avg_token_vecs(self.device, self.embedding_layer, mstr_token_seqs)
        # cat_output = torch.cat((context_lstm_output, name_output), dim=1)

        cat_output = torch.cat((context_lstm_output, name_output, entity_vecs), dim=1)
        # logits = self.linear_map(F.dropout(cat_output, self.dropout, training))
        if not self.use_mlp:
            mention_reps = self.linear_map(F.dropout(cat_output, self.dropout, training))
        else:
            l1_output = self.linear_map1(F.dropout(cat_output, self.dropout, training))
            # l1_output = F.relu(l1_output)
            l1_output = self.lin1_bn(F.relu(l1_output))
            mention_reps = self.linear_map2(F.dropout(l1_output, self.dropout, training))

        logits = torch.matmul(mention_reps.view(-1, 1, self.type_embed_dim),
                              self.type_embeddings.view(-1, self.type_embed_dim, self.n_types))
        logits = logits.view(-1, self.n_types)
        return logits


class TANEFAtt(aaa.AAA):
    def __init__(self, device, type_vocab, type_id_dict, embedding_layer: nn.Embedding, context_lstm_hidden_dim,
                 type_embed_dim, dropout=0.5, use_mlp=False, pred_mlp_hidden_dim=None, att_mlp_hidden_dim=None):
        super(TANEFAtt, self).__init__(device, type_vocab, type_id_dict, embedding_layer, context_lstm_hidden_dim,
                                       type_embed_dim, dropout)
        self.n_maps = 3
        self.use_mlp = use_mlp

        self.dropout_layer = nn.Dropout(dropout)
        linear_map_input_dim = 2 * self.context_lstm_hidden_dim + self.word_vec_dim + self.n_types
        if not self.use_mlp:
            self.linear_map = nn.Linear(linear_map_input_dim, type_embed_dim, bias=False)
            # self.linear_map = nn.Linear(linear_map_input_dim, self.n_types, bias=False)
        else:
            # mlp_hidden_dim = linear_map_input_dim // 2
            pred_mlp_hdim = linear_map_input_dim // 2 if (
                    pred_mlp_hidden_dim is None) else pred_mlp_hidden_dim
            self.lin1s = nn.ModuleList([nn.Linear(linear_map_input_dim, pred_mlp_hdim) for i in range(self.n_maps)])
            self.lin1_bns = nn.ModuleList([nn.BatchNorm1d(pred_mlp_hdim) for i in range(self.n_maps)])
            self.lin2s = nn.ModuleList([nn.Linear(pred_mlp_hdim, type_embed_dim) for i in range(self.n_maps)])

            att_mlp_input_dim = 2 * self.context_lstm_hidden_dim + self.word_vec_dim + self.n_types + 1
            att_mlp_hdim = att_mlp_input_dim // 2 if att_mlp_hidden_dim is None else att_mlp_hidden_dim
            self.att_lin1 = nn.Linear(att_mlp_input_dim, att_mlp_hdim)
            self.att_lin1_bn = nn.BatchNorm1d(att_mlp_hdim)
            self.att_lin2 = nn.Linear(att_mlp_hdim, self.n_maps)

    def forward(self, context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_sgns, el_probs):
        batch_size = len(context_token_seqs)

        context_token_seqs, seq_lens, mention_token_idxs, back_idxs = modelutils.get_len_sorted_context_seqs_input(
            self.device, context_token_seqs, mention_token_idxs)
        # self.context_hidden = self.init_context_hidden(batch_size)
        context_lstm_output = self.get_context_lstm_output(
            context_token_seqs, seq_lens, mention_token_idxs, batch_size)
        context_lstm_output = context_lstm_output[back_idxs]

        name_output = modelutils.get_avg_token_vecs(self.device, self.embedding_layer, mstr_token_seqs)
        # cat_output = torch.cat((context_lstm_output, name_output), dim=1)

        cat_output = torch.cat((context_lstm_output, name_output, entity_vecs), dim=1)
        # logits = self.linear_map(F.dropout(cat_output, self.dropout, training))
        channel_logits = list()
        att_l2 = None
        if not self.use_mlp:
            mention_reps = self.linear_map(self.dropout_layer(cat_output))
            logits = torch.matmul(mention_reps.view(-1, 1, self.type_embed_dim),
                                  self.type_embeddings.view(-1, self.type_embed_dim, self.n_types))
            logits = logits.view(-1, self.n_types)
        else:
            # print(name_output.size(), el_probs.size())
            cat_output_att = torch.cat((context_lstm_output, name_output, entity_vecs, el_probs.view(-1, 1)), dim=1)
            att_l1 = self.att_lin1(self.dropout_layer(cat_output_att))
            att_l1 = self.att_lin1_bn(F.relu(att_l1))
            att_l2 = F.softmax(self.att_lin2(self.dropout_layer(att_l1)), dim=1)

            mention_reps = 0
            logits = 0
            for i in range(self.n_maps):
                l1_output = self.lin1s[i](self.dropout_layer(cat_output))
                # l1_output = F.relu(l1_output)
                l1_output = self.lin1_bns[i](F.relu(l1_output))
                cur_mention_reps = self.lin2s[i](self.dropout_layer(l1_output))
                cur_logits = torch.matmul(cur_mention_reps.view(-1, 1, self.type_embed_dim),
                                          self.type_embeddings.view(-1, self.type_embed_dim, self.n_types))
                cur_logits = cur_logits.view(-1, self.n_types)
                channel_logits.append(cur_logits)
                logits += att_l2[:, i].view(-1, 1) * cur_logits
                # mention_reps += att_l2[:, i].view(-1, 1) * cur_mention_reps
            # logits = torch.matmul(mention_reps.view(-1, 1, self.type_embed_dim),
            #                       self.type_embeddings.view(-1, self.type_embed_dim, self.n_types))
            # logits = logits.view(-1, self.n_types)

        return logits, att_l2, channel_logits


class NEFAAATest(aaa.AAA):
    def __init__(self, device, type_vocab, type_id_dict, embedding_layer: nn.Embedding, context_lstm_hidden_dim,
                 type_embed_dim, dropout=0.5, use_mlp=False, mlp_hidden_dim=None):
        super(NEFAAATest, self).__init__(device, type_vocab, type_id_dict, embedding_layer, context_lstm_hidden_dim,
                                         type_embed_dim, dropout)
        self.linear_att = None
        self.use_mlp = use_mlp
        self.dropout_layer = nn.Dropout(dropout)

        linear_map_input_dim = 2 * self.context_lstm_hidden_dim + self.word_vec_dim + self.n_types + 1
        if not self.use_mlp:
            self.linear_map = nn.Linear(linear_map_input_dim, type_embed_dim, bias=False)
            # self.linear_map = nn.Linear(linear_map_input_dim, self.n_types, bias=False)
        else:
            mlp_hidden_dim = linear_map_input_dim // 2 if mlp_hidden_dim is None else mlp_hidden_dim
            self.linear_map1 = nn.Linear(linear_map_input_dim, mlp_hidden_dim)
            self.lin1_bn = nn.BatchNorm1d(mlp_hidden_dim)
            # self.linear_map2 = nn.Linear(mlp_hidden_dim, type_embed_dim)
            self.linear_map2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
            self.lin2_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map3 = nn.Linear(mlp_hidden_dim, type_embed_dim)

    def forward(self, context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_sgns,
                el_probs, debug_flg=False):
        batch_size = len(context_token_seqs)

        context_token_seqs, seq_lens, mention_token_idxs, back_idxs = modelutils.get_len_sorted_context_seqs_input(
            self.device, context_token_seqs, mention_token_idxs)
        # self.context_hidden = self.init_context_hidden(batch_size)
        context_lstm_output = self.get_context_lstm_output(
            context_token_seqs, seq_lens, mention_token_idxs, batch_size)
        context_lstm_output = context_lstm_output[back_idxs]

        name_output = modelutils.get_avg_token_vecs(self.device, self.embedding_layer, mstr_token_seqs)
        # cat_output = torch.cat((context_lstm_output, name_output), dim=1)

        cat_output = torch.cat((context_lstm_output, name_output, entity_vecs, el_probs.view(-1, 1)), dim=1)
        # logits = self.linear_map(F.dropout(cat_output, self.dropout, training))
        if not self.use_mlp:
            mention_reps = self.linear_map(self.dropout_layer(cat_output))
        else:
            l1_output = self.linear_map1(self.dropout_layer(cat_output))
            # l1_output = F.relu(l1_output)
            l1_output = self.lin1_bn(F.relu(l1_output))
            # mention_reps = self.linear_map2(F.dropout(l1_output, self.dropout, training))
            l2_output = self.linear_map2(self.dropout_layer(l1_output))
            l2_output = self.lin2_bn(F.relu(l2_output))
            mention_reps = self.linear_map3(self.dropout_layer(l2_output))

        logits = torch.matmul(mention_reps.view(-1, 1, self.type_embed_dim),
                              self.type_embeddings.view(-1, self.type_embed_dim, self.n_types))
        logits = logits.view(-1, self.n_types)
        return logits


class NEFAAATestStack(aaa.ResAAA):
    def __init__(self, device, type_vocab, type_id_dict, embedding_layer: nn.Embedding, context_lstm_hidden_dim,
                 type_embed_dim, dropout=0.5, use_mlp=False, mlp_hidden_dim=None, concat_lstm=False):
        super(NEFAAATestStack, self).__init__(device, type_vocab, type_id_dict, embedding_layer,
                                              context_lstm_hidden_dim, type_embed_dim, dropout, concat_lstm)
        self.linear_att = None
        self.use_mlp = use_mlp
        # self.dropout_layer = nn.Dropout(dropout)

        linear_map_input_dim = 2 * self.context_lstm_hidden_dim + self.word_vec_dim + self.n_types + 1
        if concat_lstm:
            linear_map_input_dim += 2 * self.context_lstm_hidden_dim
        if not self.use_mlp:
            self.linear_map = nn.Linear(linear_map_input_dim, type_embed_dim, bias=False)
            # self.linear_map = nn.Linear(linear_map_input_dim, self.n_types, bias=False)
        else:
            mlp_hidden_dim = linear_map_input_dim // 2 if mlp_hidden_dim is None else mlp_hidden_dim
            self.linear_map1 = nn.Linear(linear_map_input_dim, mlp_hidden_dim)
            self.lin1_bn = nn.BatchNorm1d(mlp_hidden_dim)
            # self.linear_map2 = nn.Linear(mlp_hidden_dim, type_embed_dim)
            self.linear_map2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
            self.lin2_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map3 = nn.Linear(mlp_hidden_dim, type_embed_dim)

    def forward(self, context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_sgns,
                el_probs, debug_flg=False):
        batch_size = len(context_token_seqs)

        context_token_seqs, seq_lens, mention_token_idxs, back_idxs = modelutils.get_len_sorted_context_seqs_input(
            self.device, context_token_seqs, mention_token_idxs)
        # self.context_hidden = self.init_context_hidden(batch_size)
        context_lstm_output = self.get_context_lstm_output(
            context_token_seqs, seq_lens, mention_token_idxs, batch_size)
        context_lstm_output = context_lstm_output[back_idxs]

        name_output = modelutils.get_avg_token_vecs(self.device, self.embedding_layer, mstr_token_seqs)
        # cat_output = torch.cat((context_lstm_output, name_output), dim=1)

        cat_output = torch.cat((context_lstm_output, name_output, entity_vecs, el_probs.view(-1, 1)), dim=1)
        # logits = self.linear_map(F.dropout(cat_output, self.dropout, training))
        if not self.use_mlp:
            mention_reps = self.linear_map(self.dropout_layer(cat_output))
        else:
            l1_output = self.linear_map1(self.dropout_layer(cat_output))
            # l1_output = F.relu(l1_output)
            l1_output = self.lin1_bn(F.relu(l1_output))
            # mention_reps = self.linear_map2(F.dropout(l1_output, self.dropout, training))
            l2_output = self.linear_map2(self.dropout_layer(l1_output))
            l2_output = self.lin2_bn(F.relu(l2_output))
            mention_reps = self.linear_map3(self.dropout_layer(l2_output))

        logits = torch.matmul(mention_reps.view(-1, 1, self.type_embed_dim),
                              self.type_embeddings.view(-1, self.type_embed_dim, self.n_types))
        logits = logits.view(-1, self.n_types)
        return logits
