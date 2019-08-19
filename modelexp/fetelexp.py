import logging
import torch.multiprocessing as mp
from utils import datautils
from modelexp import exputils
from models.fetentvecutils import ELDirectEntityVec
from models.feteldeep import FETELStack


def train_fetel(device, gres: exputils.GlobalRes, el_entityvec: ELDirectEntityVec,
                train_samples_pkl, dev_samples_pkl, test_mentions_file, test_sents_file, noel_preds_file,
                type_embed_dim, context_lstm_hidden_dim, learning_rate, batch_size, n_iter,
                dropout, rand_per, per_penalty, use_mlp=False, pred_mlp_hdim=None, att_mlp_hdim=None,
                save_model_file=None, nil_rate=0.5, single_type_path=False,
                stack_lstm=False, concat_lstm=False, results_file=None):
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
            # model = NEFAAATest(
            #     device, gres.type_vocab, gres.type_id_dict, gres.embedding_layer, context_lstm_hidden_dim,
            #     type_embed_dim=type_embed_dim, dropout=dropout, use_mlp=use_mlp, mlp_hidden_dim=pred_mlp_hdim)
            model = None
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


def data_producer(data_queue: mp.Queue, response_queue: mp.Queue, mpevent, el_entityvec, samples_pkl,
                  mention_token_id, parent_type_ids_dict, batch_size, n_iter):
    samples = datautils.load_pickle_data(samples_pkl)
    print('{} loaded. producer ready.'.format(samples_pkl), flush=True)
    # first_sent_model = FirstSentModel.from_file(True, first_sent_model_file, embed_layer, n_types)
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
