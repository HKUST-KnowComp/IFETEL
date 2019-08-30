import numpy as np
import json
import config
from utils import datautils


def gen_training_data_from_wiki(typed_mentions_file, sents_file, word_vecs_pkl, sample_rate,
                                n_dev_samples, output_files_name_prefix, core_title_wid_file=None):
    np.random.seed(config.RANDOM_SEED)

    core_wids = None
    if core_title_wid_file is not None:
        df = datautils.load_csv(core_title_wid_file)
        core_wids = {wid for _, wid in df.itertuples(False, None)}

    token_vocab, token_vecs = datautils.load_pickle_data(word_vecs_pkl)
    token_id_dict = {t: i for i, t in enumerate(token_vocab)}
    unknown_token_id = token_id_dict[config.TOKEN_UNK]

    f_mention = open(typed_mentions_file, encoding='utf-8')
    f_sent = open(sents_file, encoding='utf-8')
    all_samples = list()
    cur_sent = json.loads(next(f_sent))
    mention_id = 0
    for i, line in enumerate(f_mention):
        if (i + 1) % 1000000 == 0:
            print(i + 1)
        # if i > 400000:
        #     break

        v = np.random.uniform()
        if v > sample_rate:
            continue

        (wid, mention_str, sent_id, pos_beg, pos_end, target_wid, type_ids
         ) = datautils.parse_typed_mention_file_line(line)
        if core_wids is not None and target_wid not in core_wids:
            continue

        mention_str = mention_str.replace('-LRB-', '(').replace('-RRB-', ')')
        while not (cur_sent['wid'] == wid and cur_sent['sent_id'] == sent_id):
            cur_sent = json.loads(next(f_sent))
        sent_tokens = cur_sent['tokens'].split(' ')
        sent_token_ids = [token_id_dict.get(token, unknown_token_id) for token in sent_tokens]

        sample = (mention_id, mention_str, pos_beg, pos_end, target_wid, type_ids, sent_token_ids)
        mention_id += 1
        all_samples.append(sample)
        # print(i, mention_str)
        # print(sent_token_ids)
        # print()
    f_mention.close()
    f_sent.close()

    dev_samples = all_samples[:n_dev_samples]
    train_samples = all_samples[n_dev_samples:]

    print('shuffling ...', end=' ', flush=True)
    rand_perm = np.random.permutation(len(train_samples))
    train_samples_shuffled = list()
    for idx in rand_perm:
        train_samples_shuffled.append(train_samples[idx])
    train_samples = train_samples_shuffled
    print('done')

    dev_mentions, dev_sents = list(), list()
    for i, sample in enumerate(dev_samples):
        mention_id, mention_str, pos_beg, pos_end, target_wid, type_ids, sent_token_ids = sample
        mention = {'mention_id': mention_id, 'span': [pos_beg, pos_end], 'str': mention_str, 'sent_id': i}
        sent = {'sent_id': i, 'text': ' '.join([token_vocab[token_id] for token_id in sent_token_ids]),
                'afet-senid': 0, 'file_id': 'null'}
        dev_mentions.append(mention)
        dev_sents.append(sent)
    datautils.save_json_objs(dev_mentions, output_files_name_prefix + '-dev-mentions.txt')
    datautils.save_json_objs(dev_sents, output_files_name_prefix + '-dev-sents.txt')

    datautils.save_pickle_data(dev_samples, output_files_name_prefix + '-dev.pkl')
    datautils.save_pickle_data(train_samples, output_files_name_prefix + '-train.pkl')


gen_training_data_from_wiki(
    config.FIGER_FILES['typed-wiki-mentions'], config.WIKI_ANCHOR_SENTS_FILE,
    config.WIKI_FETEL_WORDVEC_FILE, 0.1, 2000, config.FIGER_FILES['anchor-train-data-prefix'])
