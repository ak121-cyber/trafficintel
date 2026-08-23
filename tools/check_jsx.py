"""Structural sanity check for app.jsx without a Babel install.

Not a parser. It catches the two mistakes hand-editing JSX actually produces:
unbalanced delimiters and unclosed elements. A syntax error in this file gives a
blank page with only a console message, so it is worth checking cheaply.
"""

import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
text = SRC.read_text(encoding="utf-8")

# ---- strip strings and comments so their contents never count as code ----
out = []
i = 0
n = len(text)
state = None            # None | "line" | "block" | quote char
while i < n:
    c = text[i]
    nxt = text[i + 1] if i + 1 < n else ""
    if state is None:
        if c == "/" and nxt == "/":
            state, i = "line", i + 2
            continue
        if c == "/" and nxt == "*":
            state, i = "block", i + 2
            continue
        if c in "\"'`":
            state, i = c, i + 1
            out.append(" ")
            continue
        out.append(c)
        i += 1
    elif state == "line":
        if c == "\n":
            state = None
            out.append("\n")
        i += 1
    elif state == "block":
        if c == "*" and nxt == "/":
            state, i = None, i + 2
            continue
        out.append("\n" if c == "\n" else " ")
        i += 1
    else:                                   # inside a string
        if c == "\\":
            i += 2
            continue
        if c == state:
            state = None
        out.append("\n" if c == "\n" else " ")
        i += 1

code = "".join(out)
if state is not None:
    print(f"FAIL: file ends inside {state!r} - unterminated string or comment")
    sys.exit(1)

# ---- delimiter balance, with line numbers for the first mismatch ----
pairs = {")": "(", "]": "[", "}": "{"}
stack = []
line = 1
for ch in code:
    if ch == "\n":
        line += 1
    elif ch in "([{":
        stack.append((ch, line))
    elif ch in ")]}":
        if not stack:
            print(f"FAIL: line {line}: closing {ch!r} with nothing open")
            sys.exit(1)
        opener, oline = stack.pop()
        if opener != pairs[ch]:
            print(f"FAIL: line {line}: {ch!r} closes {opener!r} opened on line {oline}")
            sys.exit(1)
if stack:
    opener, oline = stack[-1]
    print(f"FAIL: {opener!r} opened on line {oline} is never closed")
    sys.exit(1)

# ---- JSX element nesting ----

# A regex cannot do this: attributes contain ">" in arrow functions and in
# comparisons like alert={(n || 0) > 0}. The tag is therefore scanned character
# by character, tracking brace depth so a ">" inside {...} is ignored.
VOID = {"br", "hr", "img", "input", "meta", "link", "source", "track", "area"}
NAME = re.compile(r"[A-Za-z][A-Za-z0-9.]*")

open_tags = []
i = 0
n = len(code)
while i < n:
    if code[i] != "<":
        i += 1
        continue
    j = i + 1
    closing = False
    if j < n and code[j] == "/":
        closing, j = True, j + 1
    m = NAME.match(code, j)
    if not m:                       # "<" used as less-than, not a tag
        i += 1
        continue
    name = m.group(0)
    j = m.end()

    depth = 0
    while j < n:
        c = code[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == ">" and depth == 0:
            break
        j += 1
    if j >= n:
        ln = code.count("\n", 0, i) + 1
        print(f"FAIL: line {ln}: tag <{name}> is never terminated by '>'")
        sys.exit(1)

    self_close = code[j - 1] == "/"
    ln = code.count("\n", 0, i) + 1
    i = j + 1

    if self_close or name.lower() in VOID:
        continue
    if closing:
        if not open_tags:
            print(f"FAIL: line {ln}: </{name}> with nothing open")
            sys.exit(1)
        oname, oline = open_tags.pop()
        if oname != name:
            print(f"FAIL: line {ln}: </{name}> does not close <{oname}> from line {oline}")
            sys.exit(1)
    else:
        open_tags.append((name, ln))

if open_tags:
    oname, oline = open_tags[-1]
    print(f"FAIL: <{oname}> opened on line {oline} is never closed")
    sys.exit(1)

print(f"OK: {SRC.name} - delimiters and JSX tags balance ({text.count(chr(10)) + 1} lines)")
