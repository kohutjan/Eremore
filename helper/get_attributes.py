
def get_attributes(obj):
    attributes = dict(vars(obj))
    attributes.pop('logger', None)
    return attributes
