import csv
import sys
import tqdm

from random import shuffle
from random import randint

dict_file = sys.argv[1]
ifile = sys.argv[2]
ofile = sys.argv[3]

limit = 500
out_of = 400000

train_file = sys.argv[4]
test_file = sys.argv[5]

dict = {}
content = {}
sort_num = []

def isLatin(s):
    ss = s.decode('utf-8')
    for x in ss:
       if ord(x) > 687:
            return False
    return True

stop = set([])
for f in ['stop_words_en.txt', 'stop_words_fr.txt', 'stop_words_es.txt']:
	for line in csv.reader(open(f), delimiter=',', quoting=csv.QUOTE_MINIMAL):
		stop.add(line[0].rstrip())

for line in tqdm.tqdm(csv.reader(open(dict_file), delimiter='\t', quoting=csv.QUOTE_NONE)):
    dict[line[0].decode('utf-8').lower()] = int(line[1])

for line in tqdm.tqdm(csv.reader(open(ifile), delimiter=',', quoting=csv.QUOTE_MINIMAL)):
    if not isLatin(line[0]):
        continue
    if len(line) > 2 and not isLatin(line[1]):
        continue
    key = line[0].decode('utf-8').lower()
    num = dict.get(key)
    if num is None:
        num = 0
    sort_num.append((line[0], num))
    desc = line[-1].split()
    changed = False
    for i in range(min(7, len(desc))):
        if desc[i].decode('utf-8').lower() == key:
            desc[i] = '<concept>'
            changed = True
            break
    if changed:
        if len(desc) < 7:
            continue
        line[-1] = ' '.join(desc)
    if content.get(line[0]) is None:
        content[line[0]] = [line]
    else:
        content[line[0]].append(line)

sort_num = sorted(sort_num, key=lambda x: x[1], reverse=True)

fp = csv.writer(open(ofile, 'w'), delimiter=',', quoting=csv.QUOTE_MINIMAL)

test_sample = []
training = []
counter = 0
for k, v in sort_num:
    target = content.get(k)
    if target is None:
        continue
    for l in content.get(k):
        fp.writerow(l)
        if counter < out_of and (l[0] not in stop):
            counter += 1
            test_sample.append(l)
        else:
            training.append(l)

#shuffle(test_sample)

dedup = set([])
fp = csv.writer(open(train_file, 'w'), delimiter=',', quoting=csv.QUOTE_MINIMAL)
fp2 = csv.writer(open(test_file, 'w'), delimiter=',', quoting=csv.QUOTE_MINIMAL)

counter = 0
draw = True
for line in test_sample:
    if counter > limit or line[0] in dedup:
        training.append(line)
    else:
        if draw:
            skipfirst = randint(0, 1)
            draw = False
        if skipfirst == 1:
            training.append(line)
            skipfirst = 0
        else:
            fp2.writerow(line)
            dedup.add(line[0])
            counter += 1
            draw = True

for line in training:
    fp.writerow(line)