# -*- coding: utf-8 -*-
"""G18: 引用重编号（D 档收尾）
==========================================================
方案（PI 已批准）：
  ① 占位符替换：[[VIEIRA]]→13 [[WANKHADE]]→14 [[TUJJAR]]→15
     [[EAGLES]]→16 [[NONGNUCH]]→17 [[CHEN]]→31
  ② 正文（References 节之前）方括号引用组内数字 13-25 → +5（18-30）；
     1-12 不变。逐数字映射（非字符串替换），天然无碰撞；
     只匹配 [数字/逗号/空格] 组，不误伤 [0.686–0.734] 等区间
     （后者含小数点/连字符，不匹配该正则）
  ③ References 节整体重建：1-12 原样 + 新 13-17 + 旧 13-25 重编号为
     18-30 + Chen=31（六条新引用题录 2026-09-23 本会话 PubMed
     esummary+efetch XML 五字段验证通过）
  ④ 校验：占位符清零；正文引用 ⊆ 1..31；References 31 条连续；
     Figure Legends 之后不得出现 [数字] 引用
数据真实性：六条新引用题录逐字来自本轮 efetch/esummary 工具输出
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

MS = "E:/TBI subtype/09_tbi_aki/manuscript/MANUSCRIPT_v0.md"
BAK = "E:/TBI subtype/09_tbi_aki/manuscript/MANUSCRIPT_v0.pre_g18.md"

MAP = {n: n + 5 for n in range(13, 26)}          # 旧13-25 → 18-30
PH = {"VIEIRA": 13, "WANKHADE": 14, "TUJJAR": 15,
      "EAGLES": 16, "NONGNUCH": 17, "CHEN": 31}  # 占位符 → 新号

NEW_REFS = {
    13: ("de Cássia Almeida Vieira R, et al. Severe traumatic brain injury and acute "
         "kidney injury patients: factors associated with in-hospital mortality and "
         "unfavorable outcomes. Brain Inj. 2024;38(2):108\u2013118. PMID: 38247393."),
    14: ("Wankhade BS, et al. Acute kidney injury in critically ill patients with "
         "traumatic brain injury: a single-center retrospective cohort study. "
         "World J Crit Care Med. 2025;14(4):110079. PMID: 41377544."),
    15: ("Tujjar O, et al. Acute kidney injury after subarachnoid hemorrhage. "
         "J Neurosurg Anesthesiol. 2017;29(2):140\u2013149. PMID: 26730982."),
    16: ("Eagles ME, et al. Acute kidney injury after aneurysmal subarachnoid "
         "hemorrhage and its effect on patient outcome: an exploratory analysis. "
         "J Neurosurg. 2020;133(3):765\u2013772. PMID: 31299650."),
    17: ("Nongnuch A, Panorchan K, Davenport A. Brain-kidney crosstalk. "
         "Crit Care. 2014;18(3):225. PMID: 25043644."),
    31: ("Chen W, et al. Impact of mannitol on mortality in patients with "
         "non-traumatic intracerebral haemorrhage and acute kidney injury: "
         "a retrospective study. Sci Rep. 2025;15(1):26612. PMID: 40695945."),
}

text = open(MS, encoding="utf-8").read()

# ---- ③ 之前的准备：切分 References 节 ----
head, sep1, rest = text.partition("## References")
refs_block, sep2, tail = rest.partition("## Figure Legends")
assert sep1 and sep2, "结构切分失败：找不到 ## References / ## Figure Legends"

# 解析旧条目 {编号: 条目文本}
old_refs = {}
for m in re.finditer(r"^(\d+)\. (.+?)(?=\n\n|\Z)", refs_block, re.S | re.M):
    old_refs[int(m.group(1))] = m.group(2).strip()
print(f"[解析] 旧 References 条目数: {len(old_refs)} (期望25)")
assert sorted(old_refs) == list(range(1, 26)), f"旧编号异常: {sorted(old_refs)}"

# ---- ① 正文数字 shift（先执行！占位符仍是 [[KEY]] 形式，不受影响）----
changes = []


def shift_group(m):
    inner = m.group(1)
    parts = re.split(r"([,\s]+)", inner)
    out, nums = [], []
    for p in parts:
        if p and p[0].isdigit():
            n = int(p)
            nums.append(n)
            out.append(str(MAP.get(n, n)))
        else:
            out.append(p)
    new_inner = "".join(out)
    new_nums = [int(x) for x in re.findall(r"\d+", new_inner)]
    if new_nums != nums:
        changes.append((m.group(0), f"[{new_inner}]"))
    return f"[{new_inner}]"


head = re.sub(r"\[([\d,\s]+)\]", shift_group, head)

print(f"[shift] 13-25→+5 变更的引用组 ({len(changes)} 处)：")
for old, new in changes:
    print(f"   {old} → {new}")

# ---- ② 占位符替换（后执行：新号 13-17/31 不在 MAP 范围，不会再被 shift）----
n_ph = 0
for key, num in PH.items():
    pat = f"[[{key}]]"
    c = head.count(pat) + tail.count(pat)
    n_ph += c
    head = head.replace(pat, f"[{num}]")
    tail = tail.replace(pat, f"[{num}]")
    print(f"[占位符] [[{key}]] → [{num}]：{c} 处")

# ---- ③ References 节重建 ----
new_refs = {}
new_refs.update({n: old_refs[n] for n in range(1, 13)})   # 1-12 原样
new_refs.update(NEW_REFS)                                  # 13-17, 31
new_refs.update({n + 5: old_refs[n] for n in range(13, 26)})  # 18-30
assert sorted(new_refs) == list(range(1, 32)), f"新编号不连续: {sorted(new_refs)}"

refs_md = "## References\n\n" + "\n\n".join(
    f"{n}. {new_refs[n]}" for n in range(1, 32)) + "\n\n"

out = head + refs_md + "## Figure Legends" + tail

# ---- ④ 校验 ----
resid = sum(out.count(f"[[{k}]]") for k in PH)
assert resid == 0, f"占位符残留 {resid}"
body_cites = sorted({int(n) for n in re.findall(r"\[([\d,\s]+)\]",
                        out.partition("## References")[0]) for n in re.findall(r"\d+", n)})
bad = [n for n in body_cites if n < 1 or n > 31]
assert not bad, f"正文引用越界: {bad}"
tail_cites = re.findall(r"\[\d+\]", out.partition("## Figure Legends")[2])
assert not tail_cites, f"Figure Legends 之后残留引用: {tail_cites}"
unused = [n for n in range(1, 32) if n not in body_cites]

# ---- 备份 + 写入 ----
if not os.path.exists(BAK):
    with open(BAK, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[备份] 原稿 → MANUSCRIPT_v0.pre_g18.md")
with open(MS, "w", encoding="utf-8") as f:
    f.write(out)

print(f"\n[校验] 占位符残留=0 ✓ | References 31 条连续 ✓ | 正文引用范围 "
      f"{body_cites[0]}–{body_cites[-1]} ✓ | 图注后引用=0 ✓")
print(f"[信息] References 中未被正文引用的编号: {unused if unused else '无'}")
print("\n=== G18 完成 → MANUSCRIPT_v0.md 引用重编号落盘 ===")
