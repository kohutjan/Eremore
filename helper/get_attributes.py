
def get_attributes(obj):
    attributes = dict(vars(obj))
    keys_to_pop = ['logger']
    for k in attributes.keys():
        if k.startswith('_'):
            keys_to_pop.append(k)
    for k in keys_to_pop:
        attributes.pop(k, None)
    return attributes
