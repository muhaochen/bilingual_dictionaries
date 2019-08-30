import xml.etree.ElementTree as ET
import tqdm
import csv

import HTMLParser
html_parser = HTMLParser.HTMLParser()

#filename='./en/2742.xml'
#output='./en/2742_parse.csv'
filename=['./en/enwiktionary-latest-pages-articles.xml', './en/enwiktionary-latest-pages-meta-current.xml']
output='./es/es_en_dict2.csv'
outputmono = './es/en_dict2.cs'

start = 5

max_para = 6

meaning_start = set(['==Adjective==', '==Noun==', '==Preposition==', '==Verb==', '==Adverb==','==adjective==', '==noun==', '==preposition==', '==verb==', '==adverb=='])
translate_start = set(['==Translations==','==translations==','==Translation==','==translation=='])

common_end = '=='
desc_start = '# '
desc_start2 = '## '
language_start = '*Spanish:'

check_redundant = set([])

slang_filt = ['|slang','|archaic','qualifier|','|qualifier','lb|en','=']

def find_between( s, first, last ):
    try:
        start = s.index( first )
        end = s.index( last, start )
        return s[start + len(first):end], start, end + len(last)
    except ValueError:
        return None, None, None

transtop = '{{trans-top|'
transbottom = '{{trans-bottom'

def clean_line( s ):
    # {{}}
    s = s.strip()
    s = html_parser.unescape(s)
    while s.find('{{') > -1:
        mid, start, end = find_between(s, '{{', '}}')
        if mid is None:
            break
        if mid.find('|') > -1:
            mid = mid.split('|')[-1]
        for w in slang_filt:
            if mid.find(w) > -1:
                mid = ""
                break
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
    return s

def clean_line_light( s ):
    while s.find("'") > -1:
        s = s.replace("'", " ")
    while s.find('"') > -1:
        s = s.replace('"', " ")
    while s.find('* ') > -1:
        s = s.replace('* ', '*')
    while s.find(': ') > -1:
        s = s.replace(': ', ':')
    while s.find('{ ') > -1:
        s = s.replace('{ ', '{')
    while s.find(' }') > -1:
        s = s.replace(' }', '}')
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

fw = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

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

fno = 0
for fname in filename:
    fno += 1
    for row in open(fname):
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
            #0 default  1 meaning + translate
            role = 0
            finish_content = True
            meanings = set([])
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
                    if line in translate_start:
                        role = 1
                        lan_dict = {}
                        got_meaning = False
                        this_meaning = None
                        continue
                elif role == 1:
                    if len(line) > len(common_end) and line[:len(common_end)] == common_end:
                        # do something
                        for lan in language_start:
                            t = lan_dict.get(lan)
                            if not t is None:
                                translations.append(t)
                        role = 0
                        continue
                    elif (not got_meaning) and len(line) > len(transtop) and line[:len(transtop)] == transtop:
                        this_meaning, _, _ = find_between(line, transtop, '}}')
                        ind += 1
                        if this_meaning is None or len(this_meaning) == 0:
                            continue
                        got_meaning = True
                        if this_meaning.find('|') > -1:
                            this_meaning = this_meaning.split('|')[0]
                        continue
                    elif got_meaning:
                        this_lan = None
                        ind += 1
                        lan = language_start
                        if len(line) > len(transbottom) and line[:len(transbottom)] == transbottom:
                            got_meaning = False
                            this_meaning = None
                            continue
                        elif len(line) > len(lan) and line[:len(lan)] == lan:
                            this_lan = lan
                            line = line[len(lan):]
                        if not this_lan is None:
                            line = clean_trans(line)
                            try:
                                this_trans, _, _ = find_between(line, '{{', '}}')
                                this_trans = this_trans.split('|')[2]
                            except:
                                continue
                            if this_trans.find('[') == -1 and this_trans.find('{') == -1 and this_trans.find('(') == -1 and lan_dict.get(this_lan) is None:
                                lan_dict[this_meaning] = this_trans
                                meanings.add(this_meaning)
                # check and write_row
            if len(meanings) > 0:
                for mm in meanings:
                    tw = lan_dict.get(mm)
                    if not tw is None:
                        #try:
                        fw.writerow([title.encode('utf-8')] + [tw.encode('utf-8')] + [mm.encode('utf-8')])
                        wrote += 1
                        #except:
                        #    error += 1
            if num % 10000 == 0:
                print "Processed",num, "wrote", wrote, "error",error