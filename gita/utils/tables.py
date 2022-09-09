# @author: Yewon Lim (ga060332@yonsei.ac.kr)
def print_table(item_names, items, tab_width=50, just='right'):
    # Create strings for printing
    def _truncate(s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s
    
    key2str = {}
    for (key, val) in zip(item_names, items):
        if hasattr(val, "__float__"):
            valstr = "%-8.3g" % val
        else:
            valstr = str(val)
        key2str[_truncate(key)] = _truncate(valstr)

    # Find max widths
    if len(key2str) == 0:
        print("WARNING: tried to write empty key-value dict")
        return
    else:
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

    # Write out the data
    dashes = "-" * (keywidth + valwidth + 7)
    lines = [dashes]
    for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
        lines.append(
            "| %s%s | %s%s |"
            % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
        )
    lines.append(dashes)
    return '\n'.join(lines)+'\n'
