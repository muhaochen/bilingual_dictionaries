# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import tqdm
import csv

#filename='./fr/3472.xml'
#output='./fr/3472_parse.csv'

filename='./fr/frwiktionary-latest-pages-articles.xml'
output='./fr/fr_en_dict.csv'
start = 0

meaning_start = set(['=={{S|nom|fr|num=1}}==', '=={{S|nom|fr|num=2}}==', '=={{S|adjectif|fr}}==', '=={{S|nom|fr}}==', '=={{S|verbe|fr}}==', "=={{S|adverbe|fr}}==",u'=={{S|préposition|fr}}==','=={{S|preposition|fr}}=='])
translate_start = set(['=={{S|traductions}}=='])

common_end = '=='
desc_start = '# '
desc_start2 = '## '
language_start = ['* {{T|en}} :']


def find_between( s, first, last ):
    try:
        start = s.index( first )
        end = s.index( last, start )
        return s[start + len(first):end], start, end + len(last)
    except ValueError:
        return None, None, None

def clean_line( s ):
    # {{}}
    s = s.strip()
    while s.find('{{') > -1:
        mid, start, end = find_between(s, '{{', '}}')
        if mid is None:
            break
        if mid.find('|') > -1:
            mid = mid.split('|')[-1]
        s = s[:start] + mid + s[end:]
    while s.find('[[') > -1:
        mid, start, end = find_between(s, '[[', ']]')
        if mid is None:
            break
        if mid.find('|') > -1:
            mid = mid.split('|')[-1]
        s = s[:start] + mid + s[end:]
    while s.find('[') > -1:
        mid, start, end = find_between(s, '[', ']')
        if mid is None:
            break
        s = s[:start] + s[end:]
    while s.find('(') > -1:
        mid, start, end = find_between(s, '(', ')')
        if mid is None:
            break
        s = s[:start] + s[end:]
    while s.find('<') > -1:
        mid, start, end = find_between(s, '<', '>')
        if mid is None:
            break
        s = s[:start] + s[end:]
    while s.find("'") > -1:
        s = s.replace("'", " ")
    while s.find('"') > -1:
        s = s.replace('"', " ")
    while s.find("  ") > -1:
        s = s.replace("  ", " ")
    if len(s) > 3 and s[:3] == "fr ":
        s = s[3:]
    return s

# change for another lan
def clean_trans( s ):
    while s.find('{{qualificatif') > -1:
        mid, start, end = find_between(s, '{{qualificatif', '}}')
        if mid is None:
            break
        s = s[:start] + s[end:]
    return s


tree = ET.parse(filename)
root = tree.getroot()
fp = open(output, 'w')

fw = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
total_num = len(root)

num = 0
for node in root[start:]:
    num += 1
    title = node[0].text.strip().encode('utf-8')
    if title.find(':') > -1:
        continue
    content = None
    try:
        content = node[3][7].text
    except:
        print "Skip",title
        pass
    if content is None:
        continue
    content = content.split('\n')
    meanings = []
    translations = []
    #0 default  1 meaning  2 translate
    role = 0
    for ind in range(len(content)):
        line = content[ind]
        line = line.strip().encode('utf-8')
        while line.find('===') > -1:
            line = line.replace('===', '==')
        while line.find('== ') > -1:
            line = line.replace('== ', '==')
        while line.find(' ==') > -1:
            line = line.replace(' ==', '==')
        if role == 0:
            ind += 1
            if line in meaning_start:
                role = 1
                continue
            elif line in translate_start:
                role = 2
                lan_dict = {}
                continue
        elif role == 1:
            if len(line) > len(common_end) and line[:len(common_end)] == common_end:
                role = 0
                continue
            if len(line) > len(desc_start) + 2 and line[:len(desc_start)] == desc_start:
                # in the middle of meaning status, before clean line
                cleaned = clean_line(line[len(desc_start):])
                if cleaned.find(' ') > -1 and len(cleaned) > 2:
                    meanings.append(cleaned)
                ind += 1
                continue
            elif len(line) > len(desc_start2) + 2 and line[:len(desc_start2)] == desc_start2:
                # in the middle of meaning status, before clean line
                cleaned = clean_line(line[len(desc_start2):])
                if cleaned.find(' ') > -1 and len(cleaned) > 2:
                    meanings.append(cleaned)
                ind += 1
                continue
        elif role == 2:
            if len(line) > len(common_end) and line[:len(common_end)] == common_end:
                # do something
                for lan in language_start:
                    t = lan_dict.get(lan)
                    print t
                    if not t is None:
                        translations.append(t)
                role = 0
                continue
            else:
                this_lan = None
                ind += 1
                for lan in language_start:
                    if len(line) > len(lan) and line[:len(lan)] == lan:
                        this_lan = lan
                        line = line[len(lan):]
                        break
                if not this_lan is None:
                    line = clean_trans(line)
                    try:
                        this_trans, _, _ = find_between(line, '{{', '}}')
                        this_trans = this_trans.split('|')[2]
                    except:
                        continue
                    if this_trans.find('[') == -1 and this_trans.find('{') == -1 and this_trans.find('(') == -1 and lan_dict.get(this_lan) is None:
                        lan_dict[this_lan] = this_trans
        # check and write_row
    if len(translations) == len(translate_start) and len(meanings) > 0:
        fw.writerow([title] + translations[:len(translate_start)] + meanings)
    if num % 10000:
        print "Processed",num,"/",total_num
        print [title], translations[:len(translate_start)], meanings
        