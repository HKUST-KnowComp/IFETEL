import numpy as np
import random
import logging
from utils import datautils


class FETEntityVec:
    def get_entity_vecs(self, *input_args):
        raise NotImplementedError


class ELDirectEntityVec:
    def __init__(self, n_types, type_to_id_dict, el_system, wid_types_file):
        self.n_types = n_types
        self.el_system = el_system
        self.rand_assign_rate = 1.1
        print('loading {} ...'.format(wid_types_file))
        logging.info('rand_assign_rate={}'.format(self.rand_assign_rate))
        self.wid_types_dict = datautils.load_wid_types_file(wid_types_file, type_to_id_dict)

    def get_entity_vecs(self, mention_strs, prev_pred_results, min_popularity=10, true_wids=None,
                        filter_by_pop=False, person_type_id=None, person_l2_type_ids=None, type_vocab=None):
        all_entity_vecs = np.zeros((len(mention_strs), self.n_types), np.float32)
        el_sgns = np.zeros(len(mention_strs), np.float32)
        probs = np.zeros(len(mention_strs), np.float32)
        candidates_list = self.el_system.link_all(mention_strs, prev_pred_results)
        # print(candidates_list)
        for i, el_candidates in enumerate(candidates_list):
            # el_candidates = self.el_system.link(mstr)
            if not el_candidates:
                continue
            wid, mstr_target_cnt, popularity = el_candidates[0]
            if filter_by_pop and popularity < min_popularity:
                continue
            types = self.wid_types_dict.get(wid, None)
            if types is None:
                continue

            probs[i] = mstr_target_cnt / (sum([cand[1] for cand in el_candidates]) + 1e-7)
            el_sgns[i] = 1
            for type_id in types:
                all_entity_vecs[i][type_id] = 1

            if person_type_id is not None and person_type_id in types and (
                    self.rand_assign_rate >= 1.0 or np.random.uniform() < self.rand_assign_rate):
                for _ in range(3):
                    rand_person_type_id = person_l2_type_ids[random.randint(0, len(person_l2_type_ids) - 1)]
                    if all_entity_vecs[i][rand_person_type_id] < 1.0:
                        all_entity_vecs[i][rand_person_type_id] = 1.0
                        break
        return all_entity_vecs, el_sgns, probs
