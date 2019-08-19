import pickle
import json


def load_wid_types_file(filename, type_to_id_dict=None):
    wid_types_dict = dict()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            types = x['types']
            if type_to_id_dict:
                types = [type_to_id_dict[l] for l in types]
            wid_types_dict[x['wid']] = types
    return wid_types_dict


def load_pickle_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_pickle_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_type_vocab(type_vocab_file):
    type_vocab = list()
    type_to_id_dict = dict()
    with open(type_vocab_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            t = line.strip()
            type_vocab.append(t)
            type_to_id_dict[t] = i
    return type_vocab, type_to_id_dict
