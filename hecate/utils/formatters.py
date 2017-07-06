def sizeof_fmt(num, suffix=''):
    if abs(num) < 1000:
        return "%s%s" % (num, suffix)
    for unit in ['K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        num /= 1000.0
        if abs(num) < 1000.0:
            return "%.2f%s%s" % (num, unit, suffix)
    num /= 1000.0
    return "%.2f%s%s" % (num, 'Y', suffix)
