import tqdm
import csv
import sys

assert(len(sys.argv) > 2)

def valid_word(k):
    if len(k) < 2:
        return False
    if ord(k[0]) < 65 or (ord(k[0]) > 90 and ord(k[0]) < 97) or (ord(k[0]) > 122 and ord(k[0]) < 128):
        return False
    for w in k:
        if ord(w) < 97 or (ord(w) > 122 and ord(w) < 128):
            return False
    return True
        

ofile = sys.argv[1]
ifile = sys.argv[2:]

fp = csv.writer(open(ofile, 'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

dict = {}
dedup = set([])

for f in ifile:
    for line in tqdm.tqdm(csv.reader(open(f), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)):
        if dict.get(line[0]) is None:
            dict[line[0]] = []
        meanings = line[1].split('|')
        for m in meanings:
            dict[line[0]].append(m)

wb = []
for k, v in tqdm.tqdm(dict.iteritems()):
    if len(k.strip()) > 1 and valid_word(k):
        for vi in v:
            if len(vi) > 10 and (vi not in dedup):
                wb.append((k, vi))
                dedup.add(vi)

wb = sorted(wb, key=lambda x: x[0])

for line in wb:
    fp.writerow(line)