def get_parent_type(t):
    p = t.rfind('/')
    if p <= 1:
        return None
    return t[:p]


def get_parent_types(t):
    parents = list()
    while True:
        p = get_parent_type(t)
        if p is None:
            return parents
        parents.append(p)
        t = p


def get_parent_type_ids_dict(type_id_dict):
    d = dict()
    for t, type_id in type_id_dict.items():
        d[type_id] = [type_id_dict[p] for p in get_parent_types(t)]
    return d
