from el import elutils


def get_linked_cnts(name_entity_cnt_list_tup):
    entry_linked_cnts_dict = dict()
    for i, tup in enumerate(zip(*name_entity_cnt_list_tup)):
        _, _, _, title, cnt = tup
        cur_cnt = entry_linked_cnts_dict.get(title, 0)
        entry_linked_cnts_dict[title] = cur_cnt + cnt
    return entry_linked_cnts_dict


class SimpleEL:
    def __init__(self, mstr_target_cnt_bisect_data: elutils.MStrTargetCntBisectData,
                 title_wid_bisect_data: elutils.TitleWidBisectData,
                 redirects_bisect_data: elutils.RedirectsBisectData,
                 entry_linked_cnts_dict, wiki_id_mid_dict=None):
        self.title_wid_bisect_data = title_wid_bisect_data
        self.redirects_bisect_data = redirects_bisect_data
        self.mstr_target_cnt_bisect_data = mstr_target_cnt_bisect_data
        self.wiki_id_mid_dict = wiki_id_mid_dict
        self.entry_linked_cnts_dict = entry_linked_cnts_dict if (
                entry_linked_cnts_dict is not None) else get_linked_cnts(self.mstr_target_cnt_bisect_data)

    @staticmethod
    def init_from_candidiate_gen_pkl(pkl_file):
        import pickle

        with open(pkl_file, 'rb') as f:
            (mstr_target_cnt_bisect_data, title_wid_bisect_data, redirects_bisect_data, entity_linked_cnts_dict
             ) = pickle.load(f)
        return SimpleEL(mstr_target_cnt_bisect_data, title_wid_bisect_data, redirects_bisect_data,
                        entity_linked_cnts_dict)
