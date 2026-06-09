#!/usr/bin/env python3
"""
check_labels.py

A complementary module checking on _internal_ reference (not to a bib, but across the doc).

Specifically, check that
all labels (\label{xyz}) in a LaTeX file
have a corresponding reference (\ref{xyz} or \eqref{xyz}).

Usage: python check_labels.py <file.tex>
       python check_labels.py <dir/>   # scans all .tex files recursively
"""

import re
import sys
from pathlib import Path


__author__ = "Mark Gotham"


def extract_labels_and_refs(tex_source: str):
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex_source))
    refs   = set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}', tex_source))
    return labels, refs


def scan_path(target: Path):
    if target.is_file():
        files = [target]
    else:
        files = sorted(target.rglob("*.tex"))

    if not files:
        print("No .tex files found.")
        sys.exit(1)

    all_labels: dict[str, list[str]] = {}   # label -> [files it appears in]
    all_refs:   set[str] = set()

    for f in files:
        src = f.read_text(encoding="utf-8", errors="replace")
        labels, refs = extract_labels_and_refs(src)
        all_refs |= refs
        for lbl in labels:
            all_labels.setdefault(lbl, []).append(str(f))

    return all_labels, all_refs, files


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_labels.py <file.tex|directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: {target} does not exist.")
        sys.exit(1)

    all_labels, all_refs, files = scan_path(target)

    print(f"Scanned {len(files)} file(s).")
    print(f"Found {len(all_labels)} unique label(s), {len(all_refs)} unique ref(s).\n")

    # Labels with no matching ref
    unreferenced = {lbl: srcs for lbl, srcs in all_labels.items() if lbl not in all_refs}

    # Refs with no matching label (dangling refs)
    dangling = all_refs - set(all_labels)

    # Duplicate labels (defined in more than one file)
    duplicates = {lbl: srcs for lbl, srcs in all_labels.items() if len(srcs) > 1}

    if not unreferenced and not dangling and not duplicates:
        print("✅ All labels are referenced, all refs resolve, no duplicates. Nice work!")
        return

    if dangling:
        print(f"❌  {len(dangling)} \\ref/s with NO matching \\label (needs fixing):")
        for ref in sorted(dangling):
            print(f"    \\ref{{{ref}}}")
        print()

    if unreferenced:
        print(f"⚠️ {len(unreferenced)} label/s with NO \\ref (doesn't necessarily fixing, but good to know):")
        for lbl, srcs in sorted(unreferenced.items()):
            print(f"    \\label{{{lbl}}}")
        print()

    if duplicates:
        print(f"⚠️ {len(duplicates)} label/s defined in MULTIPLE files (doesn't necessarily fixing, but good to know):")
        for lbl, srcs in sorted(duplicates.items()):
            print(f"    \\label{{{lbl}}} found in {', '.join(srcs)}")
        print()


if __name__ == "__main__":
    main()
