from collections import namedtuple
from bisect import bisect_left

TitleWidBisectData = namedtuple('TitleWidBisectData', ['titles', 'wids'])
MStrTargetCntBisectData = namedtuple(
    'MStrTargetCntBisectData', ['mstrs', 'beg_positions', 'end_positions', 'wids', 'cnts'])
RedirectsBisectData = namedtuple('RedirectsBisectData', ['titles_from', 'wids'])


def get_mstr_targets(data: MStrTargetCntBisectData, mstr):
    mstr_idx = bisect_left(data.mstrs, mstr)
    if mstr_idx >= len(data.mstrs) or data.mstrs[mstr_idx] != mstr:
        return None

    beg_pos, end_pos = data.beg_positions[mstr_idx], data.end_positions[mstr_idx]
    return data.wids[beg_pos:end_pos], data.cnts[beg_pos: end_pos]


def get_redirected_wid(data: RedirectsBisectData, title):
    idx = bisect_left(data.titles_from, title)
    if idx >= len(data.titles_from) or data.titles_from[idx] != title:
        return None
    return data.wids[idx]


def get_wid_by_title(data: TitleWidBisectData, title):
    title_idx = bisect_left(data.titles, title)
    return data.wids[title_idx] if (title_idx < len(data.titles) and data.titles[title_idx] == title) else None
