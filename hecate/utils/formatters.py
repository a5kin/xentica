def sizeof_fmt(num, suffix=''):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            return "%3.2f%s%s" % (num, unit, suffix)
        num /= 1000.0
    return "%.2f%s%s" % (num, 'Y', suffix)
