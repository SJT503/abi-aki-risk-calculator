#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""INDEPENDENT VERIFICATION T5: manuscript / SI / Chinese docx consistency."""
import re
import sys
from docx import Document

sys.stdout.reconfigure(encoding='utf-8')
MD = r'E:/TBI subtype/09_tbi_aki/_rewrite_r12/MANUSCRIPT_npjDM_v5.md'
SI = r'E:/TBI subtype/09_tbi_aki/_rewrite_r12/Supplementary_Information_v5.md'
DX = r'E:/TBI subtype/09_tbi_aki/_rewrite_r12/MANUSCRIPT_v5_中文审阅版.docx'

md = open(MD, encoding='utf-8').read()
si = open(SI, encoding='utf-8').read()

print('='*80); print('T5a  Clinical outcomes section position'); print('='*80)
i_dep = md.find('### Deployment views')
i_out = md.find('### Clinical outcomes')
i_rob = md.find('### Robustness and subgroups')
print('Deployment@%d  Clinical outcomes@%d  Robustness@%d' % (i_dep, i_out, i_rob))
print('a)', 'MATCH' if -1 < i_dep < i_out < i_rob else '*** MISMATCH ***')

print(); print('='*80); print('T5b  supplementary-table first-mention order'); print('='*80)
mentions = re.findall(r'Supplementary Table (\d+)', md)
first = []
for m in mentions:
    n = int(m)
    if n not in first:
        first.append(n)
print('first-mention sequence:', first)
print('b)', 'MATCH' if first == [1,2,3,4,5,6,7,8] else '*** MISMATCH ***')
allm = [int(m) for m in mentions]
print('   all mention tokens:', allm)

print(); print('='*80); print('T5c  abstract phrasing'); print('='*80)
abs_m = re.search(r'## Abstract\n\n(.*?)\n\n## ', md, re.S)
abst = abs_m.group(1)
print('c1 contains "four times the in-hospital mortality":',
      'four times the in-hospital mortality' in abst,
      '->', 'MATCH' if 'four times the in-hospital mortality' in abst else '*** MISMATCH ***')
print('c2 contains "were merged":', 'were merged' in abst,
      '->', 'MATCH' if 'were merged' not in abst else '*** MISMATCH ***')

print(); print('='*80); print('T5d  abstract token count (letter-words) <= 150'); print('='*80)
toks = abst.split()
letter_words = [t for t in toks if re.search(r'[A-Za-z]', t)]
print('raw tokens:', len(toks), ' letter-words:', len(letter_words))
print('d)', 'MATCH' if len(letter_words) <= 150 else '*** MISMATCH ***')
if len(letter_words) > 150:
    print('   over-limit words sample:', letter_words[145:])

print(); print('='*80); print('T5e  Code availability'); print('='*80)
ca = re.search(r'\*\*Code availability\.\*\*(.*?)(?=\n\n\*\*)', md, re.S).group(1)
print(ca.strip()[:400])
e1 = 'archived' in ca and ('model objects' in ca or 'model objects' in ca.lower().replace('model objects','model objects'))
e1 = 'model objects' in ca and 'archived' in ca
e2 = 'fixed seed 42 inside them' not in md
print('e1 mentions model objects archived:', e1)
print('e2 "fixed seed 42 inside them" absent from whole ms:', e2)
print('e)', 'MATCH' if e1 and e2 else '*** MISMATCH ***')

print(); print('='*80); print('T5f  weakest-subgroup sentence'); print('='*80)
m = re.search(r'lowest point estimates were ([^.]*?)\.', md)
print(m.group(0) if m else 'NOT FOUND')
f1 = bool(re.search(r'0\.739 \(0\.715–0\.762\)[^.]*without chronic kidney disease', md))
f2 = bool(re.search(r'0\.752 \(0\.732–0\.775\)[^.]*aged 65 years or older', md))
print('f1 0.739 no-CKD MIMIC:', f1, ' f2 0.752 age>=65 eICU:', f2)
print('f)', 'MATCH' if f1 and f2 else '*** MISMATCH ***')

print(); print('='*80); print('T5g  SI supplementary tables'); print('='*80)
heads = re.findall(r'^#+\s*.*Supplementary Table\s*(\d+)[.:]\s*(.*)$', si, re.M)
print('SI table headings found:', heads)
nums = [int(h[0]) for h in heads]
ok_heads = nums == list(range(1, 9)) and all(nums.count(n) == 1 for n in nums)
print('g1 headings 1..8 exactly once each, ascending:', 'MATCH' if ok_heads else '*** MISMATCH ***')
hmap = {int(a): b for a, b in heads}
print('g2 Table 6 = Clinical outcomes:', hmap.get(6, '')[:40],
      '->', 'MATCH' if 'Clinical outcomes' in hmap.get(6, '') else '*** MISMATCH ***')
print('g3 Table 7 = Domain ablation:', hmap.get(7, '')[:40],
      '->', 'MATCH' if 'Domain ablation' in hmap.get(7, '') or 'ablation' in hmap.get(7, '').lower() else '*** MISMATCH ***')
print('g4 Table 8 = benchmark:', hmap.get(8, '')[:60],
      '->', 'MATCH' if 'enchmark' in hmap.get(8, '') else '*** MISMATCH ***')
# every heading occurrence count of "Supplementary Table N" as heading lines
for n in range(1, 9):
    cnt = len(re.findall(r'^#+.*Supplementary Table %d\b' % n, si, re.M))
    if cnt != 1:
        print('   *** heading Table %d appears %d times' % (n, cnt))
# contents line
cont = re.findall(r'(?i)contents[^\n]*', si)
print('contents lines:', cont[:2])
ok_cont = any(re.search(r'Tables?\s*1[–\-—]8|1\s*[–\-—]\s*8', c) for c in cont)
print('g5 Contents says Tables 1-8:', 'MATCH' if ok_cont else '*** MISMATCH ***')

print(); print('='*80); print('T5h  Chinese docx'); print('='*80)
doc = Document(DX)
paras = doc.paragraphs
texts = [p.text for p in paras]
joined = '\n'.join(texts)
print('total paragraphs:', len(paras), '(claim 91) ->',
      'MATCH' if len(paras) == 91 else '*** MISMATCH ***')
empty = [i for i, t in enumerate(texts) if t.strip() == '']
print('empty paragraphs:', len(empty), '->', 'MATCH' if not empty else '*** MISMATCH *** (idx %s)' % empty[:10])
for probe in ['临床结局', '38.9%', '补充材料Table 8将这些系统', '可精确复现冻结的外部预测']:
    c = joined.count(probe)
    print('probe %-30r count=%d ->' % (probe, c), 'MATCH' if c >= 1 else '*** MISMATCH ***')
