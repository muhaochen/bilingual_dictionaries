import tqdm
import sys
import io

ifile = sys.argv[1]
ofile = sys.argv[2]

dict = {}

for line in tqdm.tqdm(open(ifile)):
    line = line[:-1].split()
    for w in line:
        if dict.get(w) is None:
            dict[w] = 1
        else:
            dict[w] += 1

record = [(k, int(v)) for k, v in dict.iteritems()]
record = [(k.decode('utf-8'), str(v)) for k,v in sorted(record, key=lambda x: x[1], reverse=True)]

with open(ofile, 'w') as fp:
    for l in record:
        fp.write(('\t'.join(l) + u'\n').encode('utf-8'))