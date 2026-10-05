#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renumber in-text citations AND the reference list to first-citation order.

Two variants, one mapping each (computed by the main session from the ACTUAL
first-citation sequences of each variant — they differ because the npjDM variant
follows Nature-family order with Methods placed AFTER Discussion, which changes
which section cites what first).

Safety design (extends the project's prior precedent scripts/g18_renumber_refs.py):
  * Bracket groups are remapped digit-by-digit; the regex matches ONLY
    [digits, commas, spaces, en/em dashes], so numeric literals containing a
    decimal point (e.g. [0.686–0.734]) can never match.
  * ABORT (before writing) if any bracket group contains a token that is not a
    plain integer — this catches range notation like [13–17], which would
    otherwise be silently mangled.
  * ABORT if the parsed reference list is not exactly 1..31.
  * Handles BOTH section orders: npjDM puts References before the figure
    legends; JMIR puts References last. Sections are located by "## " headings,
    never by assuming a fixed neighbour.
"""
import re

MAPPING = {
    "npjDM": {"1": 1, "11": 2, "12": 3, "13": 4, "14": 5, "15": 6, "16": 7, "17": 8,
              "2": 9, "3": 10, "18": 11, "19": 12, "20": 13, "25": 14, "8": 15,
              "24": 16, "29": 17, "30": 18, "28": 19, "31": 20, "7": 21, "6": 22,
              "9": 23, "10": 24, "23": 25, "27": 26, "21": 27, "26": 28, "22": 29,
              "4": 30, "5": 31},
    "JMIR":  {"1": 1, "11": 2, "12": 3, "13": 4, "14": 5, "15": 6, "16": 7, "17": 8,
              "2": 9, "3": 10, "18": 11, "19": 12, "20": 13, "9": 14, "10": 15,
              "23": 16, "27": 17, "6": 18, "21": 19, "26": 20, "7": 21, "22": 22,
              "4": 23, "5": 24, "25": 25, "8": 26, "24": 27, "29": 28, "30": 29,
              "28": 30, "31": 31},
}

PATHS = {
    "npjDM": r"E:\TBI subtype\09_tbi_aki\manuscript\submission_npjDM\MANUSCRIPT_npjDM.md",
    "JMIR":  r"E:\TBI subtype\09_tbi_aki\manuscript\submission_JMIR\MANUSCRIPT_JMIR.md",
}

GROUP = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]")
HEAD = re.compile(r"^## ", re.M)


def split_sections(text):
    """Return (preamble, [(heading, body), ...]) split on '## ' lines."""
    marks = [m.start() for m in HEAD.finditer(text)]
    if not marks:
        return text, []
    pre = text[:marks[0]]
    secs = []
    for i, s in enumerate(marks):
        e = marks[i + 1] if i + 1 < len(marks) else len(text)
        chunk = text[s:e]
        head = chunk.split("\n", 1)[0].strip()
        secs.append((head, chunk))
    return pre, secs


for tag, path in PATHS.items():
    text = open(path, encoding="utf-8").read()
    pre, secs = split_sections(text)
    heads = [h for h, _ in secs]
    if "## References" not in heads:
        raise SystemExit(f"ABORT {tag}: no '## References' section found")

    # body = everything outside the References section and outside the legend sections
    legend_heads = {"## Figure Legends", "## Supplementary Figure Legends"}
    body_parts, ref_block = [pre], None
    for h, chunk in secs:
        if h == "## References":
            ref_block = chunk
        elif h in legend_heads:
            continue
        else:
            body_parts.append(chunk)
    body = "".join(body_parts)

    # --- guard 1: every bracket group must be pure integers/commas ---
    for g in GROUP.findall(body):
        for tok in re.split(r"[,]", g):
            tok = tok.strip()
            if tok and not tok.isdigit():
                raise SystemExit(f"ABORT {tag}: unmappable bracket group [{g}]")

    # --- guard 2: every cited number must exist in the mapping ---
    cited = {t.strip() for g in GROUP.findall(body) for t in re.split(r"[,]", g) if t.strip()}
    unknown = sorted(cited - set(MAPPING[tag]))
    if unknown:
        raise SystemExit(f"ABORT {tag}: citations not in mapping: {unknown}")

    def remap(m):
        inner = m.group(1)
        return "[" + ",".join(str(MAPPING[tag][t.strip()])
                              for t in inner.split(",")) + "]"

    new_body = GROUP.sub(remap, body)

    # --- rebuild reference list in new order ---
    entries, cur = {}, None
    for line in ref_block.split("\n")[1:]:          # skip the "## References" heading
        m = re.match(r"^(\d+)\.\s", line)
        if m:
            cur = int(m.group(1))
            entries[cur] = [line]
        elif cur is not None and line.strip():
            entries[cur].append(line)
    if sorted(entries) != list(range(1, 32)):
        raise SystemExit(f"ABORT {tag}: parsed {len(entries)} refs, expected 31")

    inv = {new: int(old) for old, new in MAPPING[tag].items()}
    rebuilt = []
    for new in range(1, 32):
        blk = list(entries[inv[new]])
        blk[0] = re.sub(r"^\d+\.", f"{new}.", blk[0], count=1)
        rebuilt += blk + [""]

    new_ref = "## References\n\n" + "\n".join(rebuilt).rstrip() + "\n\n"

    # --- reassemble, preserving the variant's own section order ---
    out_parts = [new_body.rstrip() + "\n\n"]
    for h, chunk in secs:
        if h == "## References":
            out_parts.append(new_ref)
        elif h in legend_heads:
            out_parts.append(chunk.rstrip() + "\n\n")
    open(path, "w", encoding="utf-8").write("".join(out_parts).rstrip() + "\n")
    print(f"{tag}: OK — {len(GROUP.findall(body))} groups remapped, 31 refs reordered")
