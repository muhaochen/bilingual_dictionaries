import xml.etree.ElementTree as ET
import tqdm
import csv

import HTMLParser
html_parser = HTMLParser.HTMLParser()

#filename='./en/2742.xml'
#output='./en/2742_parse.csv'
filename=['./en/enwiktionary-latest-pages-articles.xml', './en/enwiktionary-latest-pages-meta-current.xml']
output='./en/fr_en_dict.csv'
outputmono = './en/en_dict.csv'

start = 5

max_para = 6

meaning_start = set(['==Adjective==', '==Noun==', '==Preposition==', '==Verb==', '==Adverb==','==adjective==', '==noun==', '==preposition==', '==verb==', '==adverb=='])
translate_start = set(['==Translations==','==translations==','==Translation==','==translation=='])

common_end = '=='
desc_start = '# '
desc_start2 = '## '
language_start = ['*French:']

slang_filt = ['|slang','|archaic','qualifier|','|qualifier','lb|en','=','|Internet','|transitive','|countable','|uncountable','|nautical','|intransitive','|medicine', '|obsolete']

def find_between( s, first, last ):
    try:
        start = s.index( first )
        end = s.index( last, start )
        return s[start + len(first):end], start, end + len(last)
    except ValueError:
        return None, None, None

def clean_line( s, slang=slang_filt ):
    # {{}}
    s = s.strip()
    s = html_parser.unescape(s)
    while s.find('{{') > -1:
        mid, start, end = find_between(s, '{{', '}}')
        if mid is None:
            break
        for w in slang:
            if mid.find(w) > -1:
                mid = ""
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
        s = s[:start] + ' ' + s[end:]
    while s.find('(') > -1:
        mid, start, end = find_between(s, '(', ')')
        if mid is None:
            break
        s = s[:start] + ' ' + s[end:]
    while s.find('<') > -1:
        mid, start, end = find_between(s, '<', '>')
        if mid is None:
            break
        s = s[:start] + ' ' + s[end:]
    while s.find("'") > -1:
        s = s.replace("'", " ")
    while s.find('"') > -1:
        s = s.replace('"', " ")
    while s.find('|') > -1:
        s = s.replace('|', " ")
    while s.find('* ') > -1:
        s = s.replace('* ', '*')
    while s.find(': ') > -1:
        s = s.replace(': ', ':')
    while s.find(' :') > -1:
        s = s.replace(' :', ':')
    while s.find("  ") > -1:
        s = s.replace("  ", " ")
    return s.strip()

def clean_line_light( s ):
    while s.find("'") > -1:
        s = s.replace("'", " ")
    while s.find('"') > -1:
        s = s.replace('"', " ")
    while s.find('* ') > -1:
        s = s.replace('* ', '*')
    while s.find(': ') > -1:
        s = s.replace(': ', ':')
    while s.find(' :') > -1:
        s = s.replace(' :', ':')
    while s.find("  ") > -1:
        s = s.replace("  ", " ")
    return s
    

# change for another lan
def clean_trans( s ):
    s = html_parser.unescape(s)
    while s.find('{{qualifier') > -1:
        mid, start, end = find_between(s, '{{qualifier', '}}')
        if mid is None:
            break
        s = s[:start] + s[end:]
    return s

def should_filt_meaning( s ):
    s = html_parser.unescape(s)
    for w in slang_filt:
        if s.find(w) > -1:
            return True
    return False


fp = open(output, 'w')
fp2 = open(outputmono, 'w')

fw = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
fw2 = csv.writer(fp2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

num = 0
error = 0
wrote = 0
got_title = False
got_text = False
skip_title = False
new_page = False
finish_content = False
content = ""
title = None
check_redundant = set([])
fno = 0
for tree in filename:
    fno += 1
    for row in open(tree):
        #num += 1
        #title = node[0].text.strip().encode('utf-8')
        row = row.lstrip()
        # new page
        if (not new_page) and row[:6] == '<page>':
            got_title = False
            reading_text = False
            got_text = False
            skip_title = False
            new_page = True
            finish_content = False
            content = ""
            title = None
            num += 1
            continue
        if new_page and (not got_title) and (not skip_title) and row[:7] == '<title>':
            title, _, _ = find_between(row, '<title>', '</title>')
            new_page = False
            if title is None:
                skip_title = True
                continue
            try:
                title = title.strip().decode('utf-8')
            except:
                error += 1
                skip_title = True
                continue
            if title.find(':') > -1:
                skip_title = True
                continue
            if fno > 1 and title in check_redundant:
                continue
            check_redundant.add(title)
            got_title = True
        if (not skip_title) and got_title and (not got_text):
            #print "debug <text ", row.find('<text '),reading_text, row[:6] + '|'
            if (not reading_text) and row[:6] == '<text ':
                reading_text == True
                content += row[row.find('>')+1:]
                continue
            else:
                # End of content
                if len(row) > 7 and row[-8:-1] == '</text>':
                    reading_text = False
                    got_text = True
                    if len(row) > 8 and len(row[:-8].lstrip()) > 0:
                        content += row[:-8]
                    continue
                elif len(row) > 1:
                    content += row
        elif (not skip_title) and got_text and (not finish_content):
            content = content.split('\n')
            meanings = []
            translations = []
            #0 default  1 meaning  2 translate
            role = 0
            finish_content = True
            for ind in range(len(content)):
                line = content[ind]
                line = line.strip().decode('utf-8')
                while line.find('===') > -1:
                    line = line.replace('===', '==')
                while line.find('== ') > -1:
                    line = line.replace('== ', '==')
                while line.find(' ==') > -1:
                    line = line.replace(' ==', '==')
                line = clean_line_light(line)
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
                        #if should_filt_meaning(line):
                        #    continue
                        cleaned = clean_line(line[len(desc_start):])
                        if cleaned.count(' ') > 1 and len(cleaned) > 10:
                            meanings.append(cleaned)
                        ind += 1
                        continue
                    elif len(line) > len(desc_start2) + 2 and line[:len(desc_start2)] == desc_start2:
                        # in the middle of meaning status, before clean line
                        #if should_filt_meaning(line):
                        #    continue
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
            if len(translations) == len(language_start) and len(meanings) > 0:
                try:
                    fw.writerow([title.encode('utf-8')] + [w.encode('utf-8') for w in translations[:len(translate_start)]] + ['|'.join([w.encode('utf-8') for w in meanings][:max_para])])
                    wrote += 1
                except:
                    error += 1
            if len(meanings) > 0:
                try:
                    fw2.writerow([title.encode('utf-8')] + ['|'.join([w.encode('utf-8') for w in meanings])])
                except:
                    pass
            if num % 10000 == 0:
                print "Processed",num, "wrote", wrote, "error",error