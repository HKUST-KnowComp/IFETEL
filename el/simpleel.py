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

    def link(self, mstr: str, title_match_weight=500, max_num_candidates=30):
        if mstr.lower().startswith('the ') and len(mstr) > 4:
            mstr = mstr[4:]
        mstr = mstr.replace(' \'s', '\'s')

        candidates = list()
        tmp = elutils.get_mstr_targets(self.mstr_target_cnt_bisect_data, mstr)
        if tmp is not None:
            wids, cnts = tmp
            for wid, cnt in zip(wids, cnts):
                popularity = self.entry_linked_cnts_dict.get(wid, 0)
                candidates.append((wid, cnt, popularity))
                if len(candidates) == max_num_candidates:
                    break

        if ' ' not in mstr and mstr[0].islower():
            if len(mstr) == 1:
                mstr = mstr.upper()
            else:
                mstr = mstr[0].upper() + mstr[1:]

        redirected_wid = elutils.get_redirected_wid(self.redirects_bisect_data, mstr)
        if redirected_wid is not None:
            wid_direct = redirected_wid
            title_match_weight = 0
        else:
            wid_direct = elutils.get_wid_by_title(self.title_wid_bisect_data, mstr)

        mid = self.wiki_id_mid_dict.get(wid_direct, None) if (
                wid_direct and self.wiki_id_mid_dict is not None) else None

        if (mid or self.wiki_id_mid_dict is None) and wid_direct is not None:
            in_candidates = False
            for i, (cand_wid, cnt, popularity) in enumerate(candidates):
                if cand_wid == wid_direct:
                    candidates[i] = (wid_direct, cnt + title_match_weight, popularity)
                    in_candidates = True
                    break

            if not in_candidates:
                popularity = self.entry_linked_cnts_dict.get(wid_direct, 0)
                candidates.append((wid_direct, title_match_weight, popularity))
            candidates.sort(key=lambda x: -x[1])
        return candidates

    def link_all(self, mstrs, preds, max_num_candidates=30):
        if preds is None:
            return [self.link(mstr, max_num_candidates=max_num_candidates) for mstr in mstrs]
        candidates_list = list()
        i = 0
        while i < len(mstrs):
            mstr = mstrs[i]
            for j in range(i - 1, -1, -1):
                if ('/person' in preds[j] or '/PERSON' in preds[j]) and mstr in mstrs[j]:
                    # print(mstr, mstrs[j])
                    candidates_list.append(candidates_list[j])
                    break

            if len(candidates_list) < i + 1:
                candidates_list.append(self.link(mstr))
            i += 1
        return candidates_list
