import torch
from torch import nn
import numpy as np
import time
import traceback
import torch.multiprocessing as mp
from typing import List
from models.feteldeep import FETELStack
from models.fetentvecutils import ELDirectEntityVec
from modelexp import exputils
from modelexp.exputils import ModelSample, anchor_samples_to_model_samples, model_samples_from_json
import logging
from utils import datautils, utils


def __get_entity_vecs_for_samples(el_entityvec: ELDirectEntityVec, samples: List[ModelSample], noel_pred_results,
                                  filter_by_pop=False):
    # return [el_firstsent.get_entity_vec(s.mention_str) for s in samples]
    mstrs = [s.mention_str for s in samples]
    prev_pred_labels = None
    if noel_pred_results is not None:
        prev_pred_labels = [noel_pred_results[s.mention_id] for s in samples]
    return el_entityvec.get_entity_vecs(mstrs, prev_pred_labels, filter_by_pop=filter_by_pop)


def __get_entity_vecs_for_mentions(el_entityvec: ELDirectEntityVec, mentions, noel_pred_results, n_types,
                                   filter_by_pop=False):
    all_entity_type_vecs = -np.ones((len(mentions), n_types), np.float32)
    all_el_sgns = np.zeros(len(mentions), np.float32)
    all_probs = np.zeros(len(mentions), np.float32)
    mention_id_to_idxs = {m['mention_id']: i for i, m in enumerate(mentions)}
    doc_mentions_dict = utils.json_objs_to_kvlistdict(mentions, 'file_id')
    for doc_id, doc_mentions in doc_mentions_dict.items():
        prev_pred_labels = [noel_pred_results[m['mention_id']] for m in doc_mentions]
        mstrs = [m['str'] for m in doc_mentions]
        entity_type_vecs, el_sgns, probs = el_entityvec.get_entity_vecs(mstrs, prev_pred_labels,
                                                                        filter_by_pop=filter_by_pop)
        # print(entity_type_vecs.shape)
        for m, vec, el_sgn, prob_vec in zip(doc_mentions, entity_type_vecs, el_sgns, probs):
            idx = mention_id_to_idxs[m['mention_id']]
            # print(vec.shape)
            all_entity_type_vecs[idx] = vec
            all_el_sgns[idx] = el_sgn
            all_probs[idx] = prob_vec
    return all_entity_type_vecs, all_el_sgns, all_probs


def data_producer(data_queue: mp.Queue, response_queue: mp.Queue, mpevent, el_entityvec, samples_pkl,
                  mention_token_id, parent_type_ids_dict, batch_size, n_iter):
    samples = datautils.load_pickle_data(samples_pkl)
    print('{} loaded. producer ready.'.format(samples_pkl), flush=True)
    n_batches = (len(samples) + batch_size - 1) // batch_size
    data_queue.put(n_batches)
    n_steps = n_iter * n_batches
    for i in range(n_steps):
        if i > 0:
            response_queue.get()
        # print('pro gen batch', time.time(), flush=True)
        batch_idx = i % n_batches
        batch_beg, batch_end = batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(samples))
        batch_model_samples = anchor_samples_to_model_samples(
            samples[batch_beg:batch_end], mention_token_id, parent_type_ids_dict)
        entity_vecs, el_sgns, el_probs = __get_entity_vecs_for_samples(
            el_entityvec, batch_model_samples, None, True)
        data_queue.put((batch_model_samples, entity_vecs, el_sgns, el_probs))
        # print('pro batch put', time.time(), flush=True)

    data_queue.put(None)
    print('producer waiting to exit ...')
    mpevent.wait()
    print('done waiting. exit.')


def train_fetel(device, gres: exputils.GlobalRes, el_entityvec: ELDirectEntityVec, train_samples_pkl,
                dev_samples_pkl, test_mentions_file, test_sents_file, noel_preds_file, type_embed_dim,
                context_lstm_hidden_dim, learning_rate, batch_size, n_iter, dropout, rand_per, per_penalty,
                use_mlp=False, pred_mlp_hdim=None, att_mlp_hdim=None, save_model_file=None, nil_rate=0.5,
                single_type_path=False, stack_lstm=False, concat_lstm=False, results_file=None):
    logging.info(
        'type_embed_dim={} cxt_lstm_hidden_dim={} pmlp_hdim={} amlp_hdim={} nil_rate={} single_type_path={}'.format(
            type_embed_dim, context_lstm_hidden_dim, pred_mlp_hdim, att_mlp_hdim, nil_rate, single_type_path))
    logging.info('rand_per={} per_pen={}'.format(rand_per, per_penalty))
    logging.info('stack_lstm={} cat_lstm={}'.format(stack_lstm, concat_lstm))

    print('starting producer ...')
    mp.set_start_method('spawn')
    data_queue = mp.Queue()
    response_queue = mp.Queue()
    mpevent = mp.Event()
    producer = mp.Process(target=data_producer, args=(
        data_queue, response_queue, mpevent, el_entityvec, train_samples_pkl,
        gres.mention_token_id, gres.parent_type_ids_dict, batch_size, n_iter))
    producer.start()

    try:
        if stack_lstm:
            model = FETELStack(
                device, gres.type_vocab, gres.type_id_dict, gres.embedding_layer, context_lstm_hidden_dim,
                type_embed_dim=type_embed_dim, dropout=dropout, use_mlp=use_mlp, mlp_hidden_dim=pred_mlp_hdim,
                concat_lstm=concat_lstm)
        else:
            model = None
        #     model = NEFAAA(
        #         device, gres.type_vocab, gres.type_id_dict, gres.embedding_layer, context_lstm_hidden_dim,
        #         type_embed_dim=type_embed_dim, dropout=dropout, use_mlp=use_mlp, mlp_hidden_dim=pred_mlp_hdim)
        if device.type == 'cuda':
            model = model.cuda(device.index)

        dev_samples = datautils.load_pickle_data(dev_samples_pkl)
        dev_samples = anchor_samples_to_model_samples(dev_samples, gres.mention_token_id, gres.parent_type_ids_dict)

        train_proc(gres, model, el_entityvec, data_queue, response_queue, dev_samples, test_mentions_file,
                   test_sents_file, noel_preds_file, nil_rate, learning_rate, n_iter, rand_per,
                   per_penalty=per_penalty, single_type_path=single_type_path, save_model_file=save_model_file,
                   results_file=results_file)
    except:
        print(traceback.format_exc())
        producer.terminate()
        producer.join()
    mpevent.set()
    print('mpevent set')


def __get_l2_person_type_ids(type_vocab):
    person_type_ids = list()
    for i, t in enumerate(type_vocab):
        if t.startswith('/person') and t != '/person':
            person_type_ids.append(i)
    return person_type_ids


def train_proc(gres: exputils.GlobalRes, model, el_entityvec: ELDirectEntityVec, data_queue, response_queue,
               dev_samples: List[ModelSample], test_mentions_file, test_sents_file, test_noel_preds_file, nil_rate,
               learning_rate, n_iter, rand_per, per_penalty, single_type_path,
               save_model_file=None, eval_batch_size=32, filter_by_pop=False, results_file=None):
    lr_gamma = 0.7
    logging.info('{}'.format(model.__class__.__name__))
    dev_true_labels_dict = {s.mention_id: [gres.type_vocab[l] for l in s.labels] for s in dev_samples}
    dev_entity_vecs, dev_el_sgns, dev_el_probs = __get_entity_vecs_for_samples(
        el_entityvec, dev_samples, None, filter_by_pop)

    test_samples = model_samples_from_json(gres.token_id_dict, gres.unknown_token_id, gres.mention_token_id,
                                           gres.type_id_dict, test_mentions_file, test_sents_file)
    test_noel_pred_results = datautils.read_pred_results_file(test_noel_preds_file)
    mentions = datautils.read_json_objs(test_mentions_file)
    test_true_labels_dict = {s.mention_id: [gres.type_vocab[l] for l in s.labels] for s in test_samples}
    test_entity_vecs, test_el_sgns, test_el_probs = __get_entity_vecs_for_mentions(
        el_entityvec, mentions, test_noel_pred_results, gres.n_types, filter_by_pop)
    person_type_id = gres.type_id_dict.get('/person')
    l2_person_type_ids = None
    person_loss_vec = None
    if person_type_id is not None:
        l2_person_type_ids = __get_l2_person_type_ids(gres.type_vocab)
        person_loss_vec = np.ones(gres.n_types, np.float32)
        for tid in l2_person_type_ids:
            person_loss_vec[tid] = per_penalty
        person_loss_vec = torch.tensor(person_loss_vec, dtype=torch.float32, device=model.device)

    # dev_results_file = '/home/hldai/data/fet/nef-results/wiki-dev-results.txt'
    dev_results_file = None
    # test_results_file = '/home/hldai/data/fet/nef-results/bbn-test-results.txt'
    n_batches = data_queue.get()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_batches, gamma=lr_gamma)
    losses = list()
    best_dev_acc = -1
    logging.info('{} steps, {} steps per iter, lr_decay={}, start training ...'.format(
        n_iter * n_batches, n_batches, lr_gamma))
    step = 0
    while True:
        batch_data = data_queue.get()
        if batch_data is None:
            break
        response_queue.put('OK')
        batch_samples, entity_vecs, el_sgns, el_probs = batch_data
        # use_entity_vecs = step > n_batches
        use_entity_vecs = True

        model.train()

        if rand_per:
            (context_token_seqs, mention_token_idxs, mstrs, mstr_token_seqs, type_vecs
             ) = exputils.get_mstr_context_batch_input_rand_per(
                model.device, gres.n_types, batch_samples, person_type_id, l2_person_type_ids)
        else:
            (context_token_seqs, mention_token_idxs, mstrs, mstr_token_seqs, type_vecs
             ) = exputils.get_mstr_context_batch_input(model.device, gres.n_types, batch_samples)

        if use_entity_vecs:
            for i in range(entity_vecs.shape[0]):
                if np.random.uniform() < nil_rate:
                    entity_vecs[i] = np.zeros(entity_vecs.shape[1], np.float32)
                    el_sgns[i] = 0
            el_sgns = torch.tensor(el_sgns, dtype=torch.float32, device=model.device)
            el_probs = torch.tensor(el_probs, dtype=torch.float32, device=model.device)
            entity_vecs = torch.tensor(entity_vecs, dtype=torch.float32, device=model.device)
        else:
            entity_vecs = None
        logits = model(context_token_seqs, mention_token_idxs, mstr_token_seqs, entity_vecs, el_sgns, el_probs)
        loss = model.get_loss(type_vecs, logits, person_loss_vec=person_loss_vec)
        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, float('inf'))
        optimizer.step()
        losses.append(loss.data.cpu().numpy())

        step += 1
        if step % 1000 == 0:
            # logging.info('i={} l={:.4f}'.format(step + 1, sum(losses)))
            l_v, acc_v, pacc_v, _, _, dev_results = eval_fetel(
                gres, model, dev_samples, dev_true_labels_dict, dev_entity_vecs, dev_el_sgns, dev_el_probs,
                eval_batch_size, use_entity_vecs=use_entity_vecs, single_type_path=single_type_path)
            _, acc_t, pacc_t, maf1, mif1, test_results = eval_fetel(
                gres, model, test_samples, test_true_labels_dict, test_entity_vecs, test_el_sgns, test_el_probs,
                eval_batch_size, use_entity_vecs=use_entity_vecs, single_type_path=single_type_path)

            best_tag = '*' if acc_v > best_dev_acc else ''
            logging.info(
                'i={} l={:.4f} l_v={:.4f} acc_v={:.4f} paccv={:.4f} acc_t={:.4f} maf1={:.4f} mif1={:.4f}{}'.format(
                    step, sum(losses), l_v, acc_v, pacc_v, acc_t, maf1, mif1, best_tag))
            if acc_v > best_dev_acc and save_model_file:
                torch.save(model.state_dict(), save_model_file)
                logging.info('model saved to {}'.format(save_model_file))

            if dev_results_file is not None and acc_v > best_dev_acc:
                datautils.save_json_objs(dev_results, dev_results_file)
                logging.info('dev reuslts saved {}'.format(dev_results_file))
            if results_file is not None and acc_v > best_dev_acc:
                datautils.save_json_objs(test_results, results_file)
                logging.info('test reuslts saved {}'.format(results_file))

            if acc_v > best_dev_acc:
                best_dev_acc = acc_v
            losses = list()


def eval_fetel(gres: exputils.GlobalRes, model, samples: List[ModelSample], true_labels_dict,
               entity_vecs, el_sgns, el_probs, batch_size=32, use_entity_vecs=True, single_type_path=False):
    model.eval()
    n_batches = (len(samples) + batch_size - 1) // batch_size
    losses = list()
    pred_labels_dict = dict()
    result_objs = list()
    for i in range(n_batches):
        batch_beg, batch_end = i * batch_size, min((i + 1) * batch_size, len(samples))
        batch_samples = samples[batch_beg:batch_end]
        (context_token_seqs, mention_token_idxs, mstrs, mstr_token_seqs, type_vecs
         ) = exputils.get_mstr_context_batch_input(model.device, gres.n_types, batch_samples)
        entity_vecs_batch, el_sgns_batch, el_probs_batch = None, None, None
        if use_entity_vecs:
            # entity_vecs, el_sgns = __get_entity_vecs_for_samples(el_entityvec, batch_samples, noel_pred_results)
            entity_vecs_batch = torch.tensor(entity_vecs[batch_beg:batch_end], dtype=torch.float32,
                                             device=model.device)
            el_sgns_batch = torch.tensor(el_sgns[batch_beg:batch_end], dtype=torch.float32, device=model.device)
            el_probs_batch = torch.tensor(el_probs[batch_beg:batch_end], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            logits = model(context_token_seqs, mention_token_idxs, mstr_token_seqs,
                           entity_vecs_batch, el_sgns_batch, el_probs_batch)
            loss = model.get_loss(type_vecs, logits)
        losses.append(loss)

        if single_type_path:
            preds = model.inference(logits)
        else:
            preds = model.inference_full(logits, extra_label_thres=0.0)
        for j, (sample, type_ids_pred, sample_logits) in enumerate(
                zip(batch_samples, preds, logits.data.cpu().numpy())):
            labels = utils.get_full_types([gres.type_vocab[tid] for tid in type_ids_pred])
            pred_labels_dict[sample.mention_id] = labels
            result_objs.append({'mention_id': sample.mention_id, 'labels': labels,
                                'logits': [float(v) for v in sample_logits]})

    strict_acc = utils.strict_acc(true_labels_dict, pred_labels_dict)
    partial_acc = utils.partial_acc(true_labels_dict, pred_labels_dict)
    maf1 = utils.macrof1(true_labels_dict, pred_labels_dict)
    mif1 = utils.microf1(true_labels_dict, pred_labels_dict)
    return sum(losses), strict_acc, partial_acc, maf1, mif1, result_objs
