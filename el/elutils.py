from collections import namedtuple

TitleWidBisectData = namedtuple('TitleWidBisectData', ['titles', 'wids'])
MStrTargetCntBisectData = namedtuple(
    'MStrTargetCntBisectData', ['mstrs', 'beg_positions', 'end_positions', 'wids', 'cnts'])
RedirectsBisectData = namedtuple('RedirectsBisectData', ['titles_from', 'wids'])
