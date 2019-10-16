import datetime
import torch
import numpy as np
import os
import logging
from utils.loggingutils import init_universal_logging
from models import fetentvecutils
from modelexp import fetelexp, exputils
from el import simpleel
import config


def train_model():
    batch_size = 256
    dropout = 0.5
    context_lstm_hidden_dim = 250
    type_embed_dim = 500
    pred_mlp_hdim = 500
    n_iter = 15
    lr = 0.001
    nil_rate = 0.5
    use_mlp = True
    rand_per = True
    stack_lstm = True
    concat_lstm = False
    per_pen = 2.0

    dataset = 'figer'
    # dataset = 'bbn'
    datafiles = config.FIGER_FILES if dataset == 'figer' else config.BBN_FILES
    single_type_path = True if dataset == 'bbn' else False
    test_mentions_file = datafiles['fetel-test-mentions']

    word_vecs_file = config.WIKI_FETEL_WORDVEC_FILE

    data_prefix = datafiles['anchor-train-data-prefix']
    dev_data_pkl = data_prefix + '-dev.pkl'
    train_data_pkl = data_prefix + '-train.pkl'

    save_model_file = None
    results_file = None
    # results_file = os.path.join(config.DATA_DIR, 'result/{}-{}.txt'.format(
    #     os.path.splitext(os.path.basename(test_mentions_file))[0], dataset))
    noel_preds_file = datafiles['noel-typing-results']

    el_candidates_file = config.EL_CANDIDATES_DATA_FILE
    print('init el with {} ...'.format(el_candidates_file), end=' ', flush=True)
    el_system = simpleel.SimpleEL.init_from_candidiate_gen_pkl(el_candidates_file)
    print('done', flush=True)

    gres = exputils.GlobalRes(datafiles['type-vocab'], word_vecs_file)

    el_entityvec = fetentvecutils.ELDirectEntityVec(
        gres.n_types, gres.type_id_dict, el_system, datafiles['wid-type-file'])

    logging.info('dataset={} {}'.format(dataset, data_prefix))
    fetelexp.train_fetel(
        device, gres, el_entityvec, train_data_pkl, dev_data_pkl, test_mentions_file, datafiles['fetel-test-sents'],
        test_noel_preds_file=noel_preds_file, type_embed_dim=type_embed_dim,
        context_lstm_hidden_dim=context_lstm_hidden_dim, learning_rate=lr, batch_size=batch_size, n_iter=n_iter,
        dropout=dropout, rand_per=rand_per, per_penalty=per_pen, use_mlp=use_mlp, pred_mlp_hdim=pred_mlp_hdim,
        save_model_file=save_model_file, nil_rate=nil_rate, single_type_path=single_type_path,
        stack_lstm=stack_lstm, concat_lstm=concat_lstm, results_file=results_file)


if __name__ == '__main__':
    import random
    torch.random.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.NP_RANDOM_SEED)
    random.seed(config.PY_RANDOM_SEED)
    str_today = datetime.date.today().strftime('%y-%m-%d')
    log_file = os.path.join(config.LOG_DIR, '{}-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], str_today, config.MACHINE_NAME))
    init_universal_logging(log_file, mode='a', to_stdout=True)

    device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')

    train_model()
