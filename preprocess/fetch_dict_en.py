import csv
#change
#ill_files = ["../ITransE/data/WK3l-15k/en_fr/en2fr_fk.csv", "../ITransE/data/WK3l-15k/en_de/en2de_fk.csv", "../ITransE/data/WK3l-120k/en_fr/en2fr_fk_120k.csv", "../ITransE/data/WK3l-120k/en_de/en2de_fk_120k.csv"]
ill_files = ["preprocess/raw_en_fr.csv"]
#change
desc_file = "short_abstracts_en.ttl"
#change
ofile = "preprocess/all/en_dict3.csv"
ofile2 = "preprocess/all/fr_en_dict3.csv"


def parse_name(name):
    s = name
    b = 0
    b = s.find('resource/')
    if b > 0:
        s = s[b+9:]
    s = s.lower()
    b = s.find(':')
    if b > 0:
        s = s[b + 1:]
    while s.find('_') > -1:
        s = s.replace('_',' ')
    while s.find('-') > -1:
        s = s.replace('-',' ')
    while s.find("'") > -1:
        s = s.replace("'",' ')
    while s.find('  ') > -1:
        s = s.replace('  ', ' ')
    return s.strip()

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end], end
    except ValueError:
        return "", -1

def find_between_extr( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.rindex( last, start )
        return s[start + len(first):end], start - len( first ), end
    except ValueError:
        return "", -1, -1

def parse_text(s):
    while True:
        _, b, e = find_between_extr(s, '(', ')')
        if b == -1:
            break
        else:
            s = s[0:b] + s[e + 1:]
    while True:
        _, b, e = find_between_extr(s, '[', ']')
        if b == -1:
            break
        else:
            s = s[0:b] + s[e + 1:]
    while True:
        _, b, e = find_between_extr(s, '{', '}')
        if b == -1:
            break
        else:
            s = s[0:b] + s[e + 1:]
    while s.find('  ') > -1:
        s = s.replace('  ', ' ')
    return s.strip()

def first_sen(s):
    e = s.find('. ')
    if e > 30:
        return s[:e+1]
    elif len(s) > 30:
        e = s[30:].find('. ')
        if e > 0:
            return s[30 + e:31+e]
    return s

def is_num(s):
    try:
        val = float(s)
        return True
    except ValueError:
        return False

#change
lan_pos = 0

num_covered = 0
num_ill = 0
num_ill_covered = 0

ill_ent = set([])
lan_map = {}

for f in ill_files:
    for line in open(f):
        line = line.rstrip('\r').rstrip('\n').split('@@@')
        if len(line) != 2:
            continue
        name = parse_text(parse_name(line[lan_pos]))
        tname = parse_text(parse_name(line[1 - lan_pos]))
        if lan_map.get(name) is None and len(tname) > 1 and tname.find(' ') == -1:
            lan_map[name] = tname
    print "Finished", f

fp = csv.writer(open(ofile, 'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
fp2 = csv.writer(open(ofile2, 'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
scanned = 0
for line in open(desc_file):
    name, e = find_between(line, '<', '>')
    line = line[e+1:]
    name = parse_name(name)
    name = parse_text(name)
    if name.find(' ') > -1 or is_num(name):
        continue
    #change
    txt, e = find_between(line, '> "', '"@en')
    txt = first_sen(parse_text(txt))
    fp.writerow([name] + [txt])
    trans = lan_map.get(name)
    if not trans is None:
        fp2.writerow([name] + [trans] + [txt])
        num_ill_covered += 1
    scanned += 1
    if scanned % 100000 == 0:
        print "scanned", scanned
        print "  current:", name

print num_ill_covered, '/', num_ill