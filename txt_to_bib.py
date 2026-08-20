#!/usr/bin/env python3
"""
txt_to_bib.py — Attempt to map a plain-text bibliography into structured BibTeX.

Plain-text reference lists come in a wide range of styles spanning:
- several standards (e.g., APA, MLA, Chicago, ...),
- numbered / unnumbered items (or otherwise prefixed e.g., with cite key)
- multiple venue types (journal articles, books, book chapters, ...).

... No regex will get every entry perfect.

Here, the approach is to:
1. Split the input text into individual reference "entries".
2. Try several style-specific patterns to pull out
    author(s), year, title, and source (journal/book/publisher/conference).
3. Form a best guess for the BibTeX entry type
    (@article, @book, @incollection, ...)
4. Build a unique citation key,
5. Write a .bib file, and flag any entry about which we're confident about

No third-party dependencies here — standard library only.

USAGE
-----
python txt_to_bib.py input.txt -o output.bib

OPTIONS
-------
`--split` with choices being ["auto", "blank", "lines", "numbered"],

E.g.,
`--split lines`: treat every non-blank line as its own entry.
`--split numbered`: If entries are known to be numbered like "1." or "[1]" at the start of each one

"""

import argparse
import re
import sys
import unicodedata
from collections import Counter


__author__ = "Mark Gotham"

# -----------------------------------------------------------------------------

# 1. Split raw text into candidate entries


def split_entries(text, mode="auto"):
    """Split the raw bibliography text into a list of entry strings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if mode == "auto":
        # Heuristic: if there are many numbered lines like "1." / "[1]" /
        # "1)" at the start of paragraphs, use numbered splitting.
        numbered_starts = re.findall(r"(?m)^\s*(?:\[\d+\]|\d{1,3}[.)])\s+", text)
        blank_sep_blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
        if len(numbered_starts) >= max(3, len(blank_sep_blocks) * 0.6):
            mode = "numbered"
        elif len(blank_sep_blocks) >= 2:
            mode = "blank"
        else:
            mode = "lines"

    if mode == "numbered":
        # Split right before each numbering marker.
        parts = re.split(r"(?m)^\s*(?:\[\d+\]|\d{1,3}[.)])\s+", text)
        entries = [p.strip() for p in parts if p.strip()]
    elif mode == "blank":
        entries = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    elif mode == "lines":
        entries = [l.strip() for l in text.split("\n") if l.strip()]
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    # Collapse internal newlines/extra whitespace within each entry.
    entries = [re.sub(r"\s+", " ", e).strip() for e in entries]
    return entries


# --------------------------------------------------------------------------

# 2. Field extraction heuristics


YEAR_RE = re.compile(r"\(?\b(1[5-9]\d{2}|20\d{2})[a-z]?\b\)?")
DOI_RE = re.compile(r"\b(?:doi:|https?://doi\.org/)\s*(10\.\d{4,9}/\S+)", re.I)
URL_RE = re.compile(r"https?://\S+")
PAGES_RE = re.compile(r"\b(?:pp?\.\s*)?(\d{1,5})\s*[-–—]\s*(\d{1,5})\b")
VOL_ISSUE_RE = re.compile(r"\b(\d{1,4})\s*\((\d{1,3})\)")
VOL_ONLY_RE = re.compile(r"\bvol(?:ume)?\.?\s*(\d{1,4})\b", re.I)


def strip_trailing_punct(s):
    return s.strip().strip(".,;: ").strip()


def extract_doi_url(entry):
    doi = DOI_RE.search(entry)
    if doi:
        return doi.group(1).rstrip(".,)"), None
    url = URL_RE.search(entry)
    if url:
        return None, url.group(0).rstrip(".,)")
    return None, None


def extract_year(entry):
    m = YEAR_RE.search(entry)
    if m:
        return m.group(1), m.span()
    return None, None


def extract_pages(entry):
    m = PAGES_RE.search(entry)
    if m:
        return f"{m.group(1)}--{m.group(2)}"
    return None


def extract_volume_issue(entry):
    m = VOL_ISSUE_RE.search(entry)
    if m:
        return m.group(1), m.group(2)
    m = VOL_ONLY_RE.search(entry)
    if m:
        return m.group(1), None
    return None, None


def extract_authors(pre_year_text):
    """
    Guess an author list from the text that appears before the year.
    Handles
    'Last, F. M., Last2, F., & Last3, G.' and
    'F. Last, F. Last2, and F. Last3' styles,
    as well as the all important 'et al.'.
    """
    txt = strip_trailing_punct(pre_year_text)
    if not txt:
        return None

    et_al = bool(re.search(r"\bet al\.?\b", txt, re.I))
    txt = re.sub(r"\bet al\.?\b", "", txt, flags=re.I).strip(", ")

    # Normalize connectors: "and", "&" -> ","
    txt = re.sub(r"\s*&\s*", ", ", txt)
    txt = re.sub(r",?\s+and\s+", ", ", txt)

    # Style A: "Lastname, F. M." repeated, comma separated -> group pairs.
    # Try splitting on ", " and re-pairing "Last" + "Initials".
    tokens = [t.strip() for t in txt.split(",") if t.strip()]

    authors = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # If next token looks like initials (short, has `.`s / single caps)
        if i + 1 < len(tokens) and re.match(r"^([A-Z]\.?\s*){1,3}$", tokens[i + 1]):
            authors.append(f"{tok}, {tokens[i + 1]}")
            i += 2
        elif re.match(r"^[A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']*$", tok):
            # "Firstname Lastname" already whole
            parts = tok.rsplit(" ", 1)
            authors.append(f"{parts[1]}, {parts[0]}")
            i += 1
        else:
            authors.append(tok)
            i += 1

    if et_al:
        authors.append("others")

    authors = [a for a in authors if a]
    if not authors:
        return None
    return " and ".join(authors)


def guess_entry_type(entry):
    low = entry.lower()
    if re.search(r"\bproceedings\b|\bconference\b|\bworkshop\b|\bsymposium\b", low):
        return "inproceedings"
    if re.search(r"\bph\.?d\.?\s+thesis\b|\bdoctoral dissertation\b|\bmaster'?s thesis\b", low):
        return "phdthesis"
    if re.search(r"\bin\s+[A-Z][\w &]+\(eds?\.?\)|\bin\s+.+,\s*eds?\.", entry):
        return "incollection"
    if re.search(r"\bpress\b|\bpublisher\b|\bpublishing\b", low) and not re.search(
        r"\bjournal\b|\bvol\.|\(\d{1,3}\)", low
    ):
        return "book"
    if URL_RE.search(entry) and not re.search(r"\bjournal\b|\bvol\.", low):
        if re.search(r"\bretrieved\b|\baccessed\b|\bwebsite\b", low):
            return "misc"
    if re.search(r"\bjournal\b", low) or VOL_ISSUE_RE.search(entry) or PAGES_RE.search(entry):
        return "article"
    return "misc"


def extract_title_and_source(after_year_text, entry_type):
    """
    Given the text after the year, split into (title, source_venue).
    Title is usually the first sentence-like chunk up to a `.`,
    unless quoted.
    """
    txt = strip_trailing_punct(after_year_text)
    if not txt:
        return None, None

    # Quoted title: "Title." Source.
    q = re.match(r'^[“"](.+?)[”"]\.?\s*(.*)$', txt)
    if q:
        return strip_trailing_punct(q.group(1)), strip_trailing_punct(q.group(2)) or None

    # Otherwise split on first `.` that's followed by a capital letter
    # or end of string (avoids splitting on "et al." / "U.S." etc.)
    parts = re.split(r"\.\s+(?=[A-Z])", txt, maxsplit=1)
    if len(parts) == 2:
        title, rest = parts
    else:
        title, rest = txt, ""

    return strip_trailing_punct(title), strip_trailing_punct(rest) or None


# --------------------------------------------------------------------------

# 3. Assemble a structured record from one entry string


def parse_entry(entry, index):
    warnings = []

    doi, url = extract_doi_url(entry)
    pages = extract_pages(entry)
    volume, number = extract_volume_issue(entry)

    # Work from a version of the entry with DOI/URL stripped out.
    # This makes sure they don't leak into title/venue text.
    cleaned = DOI_RE.sub("", entry)
    cleaned = URL_RE.sub("", cleaned)

    year, year_span = extract_year(cleaned)
    if not year:
        warnings.append("no year found")
        pre_year, post_year = cleaned, ""
    else:
        pre_year = cleaned[: year_span[0]]
        post_year = cleaned[year_span[1]:]

    authors = extract_authors(pre_year)
    if not authors:
        warnings.append("could not confidently parse author(s)")

    entry_type = guess_entry_type(entry)
    title, source = extract_title_and_source(post_year, entry_type)

    # Strip volume/issue/pages fragments and stray punctuation out of the
    # venue text, since they're captured as their own fields already.
    if source:
        source = VOL_ISSUE_RE.sub("", source)
        source = VOL_ONLY_RE.sub("", source)
        source = PAGES_RE.sub("", source)
        source = re.sub(r"\bRetrieved from\b", "", source, flags=re.I)
        # Clean up punctuation debris left behind by the substitutions above
        # (e.g. "Journal Name, , ." -> "Journal Name").
        source = re.sub(r"\s*,\s*(?=,|\.|$)", "", source)
        source = re.sub(r"[,\s]+$", "", source)
        source = re.sub(r"\(\s*\)", "", source)  # empty parens left by pages removal
        source = re.sub(r"^\s*In\s+", "", source)  # "In Proceedings of..." -> "Proceedings of..."
        source = re.sub(r"\s{2,}", " ", source).strip()
        source = strip_trailing_punct(source)
        source = source or None

    if not title:
        warnings.append("could not confidently parse title")

    record = {
        "type": entry_type,
        "authors": authors,
        "year": year,
        "title": title,
        "source": source,
        "volume": volume,
        "number": number,
        "pages": pages,
        "doi": doi,
        "url": url,
        "raw": entry,
        "warnings": warnings,
        "index": index,
    }
    return record


# --------------------------------------------------------------------------

# 4. Citation key generation


def slugify_word(w):
    w = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]", "", w)


def make_key(record, used_keys):
    author_part = "Unknown"
    if record["authors"]:
        first_author = record["authors"].split(" and ")[0]
        last = first_author.split(",")[0]
        author_part = slugify_word(last) or "Unknown"

    year_part = record["year"] or "n.d."
    title_word = ""
    if record["title"]:
        stop = {"a", "an", "the", "on", "of", "in", "and", "for", "to"}
        for w in record["title"].split():
            sw = slugify_word(w)
            if sw and sw.lower() not in stop:
                title_word = sw
                break

    base = f"{author_part}{year_part}{title_word}"
    key = base
    n = 1
    while key in used_keys:
        n += 1
        key = f"{base}{chr(96 + n)}"  # a, b, c...
    used_keys.add(key)
    return key


# --------------------------------------------------------------------------

# 5. BibTeX rendering


FIELD_ORDER = ["author", "title", "journal", "booktitle", "publisher",
               "school", "year", "volume", "number", "pages", "doi",
               "url", "note"]


def escape_braces(s):
    if s is None:
        return s
    return s.replace("{", "\\{").replace("}", "\\}")


def build_fields(record):
    fields = {}
    if record["authors"]:
        fields["author"] = record["authors"]
    if record["title"]:
        fields["title"] = record["title"]
    if record["year"]:
        fields["year"] = record["year"]
    if record["volume"]:
        fields["volume"] = record["volume"]
    if record["number"]:
        fields["number"] = record["number"]
    if record["pages"]:
        fields["pages"] = record["pages"]
    if record["doi"]:
        fields["doi"] = record["doi"]
    if record["url"]:
        fields["url"] = record["url"]

    src = record["source"]
    etype = record["type"]
    if src:
        if etype == "article":
            fields["journal"] = src
        elif etype in ("inproceedings",):
            fields["booktitle"] = src
        elif etype == "book":
            fields["publisher"] = src
        elif etype == "phdthesis":
            fields["school"] = src
        elif etype == "incollection":
            fields["booktitle"] = src
        else:
            # Unclear venue type; keep it visible without guessing wrong.
            fields.setdefault("note", src)

    if record["warnings"]:
        note = "NEEDS REVIEW: " + "; ".join(record["warnings"])
        fields["note"] = (fields.get("note", "") + " | " + note).strip(" |") \
            if fields.get("note") else note

    return fields


def render_entry(key, record):
    fields = build_fields(record)
    lines = [f"@{record['type']}{{{key},"]
    ordered_keys = [f for f in FIELD_ORDER if f in fields]
    ordered_keys += [f for f in fields if f not in FIELD_ORDER]
    for i, fname in enumerate(ordered_keys):
        val = escape_braces(fields[fname])
        comma = "," if i < len(ordered_keys) - 1 else ""
        lines.append(f"  {fname} = {{{val}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------

# Main driver (outside 1-5)


def convert(text, split_mode="auto"):
    raw_entries = split_entries(text, split_mode)
    records = [parse_entry(e, i) for i, e in enumerate(raw_entries, 1)]

    used_keys = set()
    rendered = []
    for r in records:
        key = make_key(r, used_keys)
        r["key"] = key
        rendered.append(render_entry(key, r))

    bib_text = "\n\n".join(rendered) + "\n"
    return bib_text, records


def print_report(records):
    n = len(records)
    flagged = [r for r in records if r["warnings"]]
    type_counts = Counter(r["type"] for r in records)

    print(f"Parsed {n} entries.", file=sys.stderr)
    print("Entry types: " + ", ".join(f"{t}={c}" for t, c in type_counts.items()),
          file=sys.stderr)
    if flagged:
        print(f"\n{len(flagged)} entries flagged for manual review:", file=sys.stderr)
        for r in flagged:
            snippet = r["raw"][:80] + ("..." if len(r["raw"]) > 80 else "")
            print(f"  [{r['key']}] {', '.join(r['warnings'])}", file=sys.stderr)
            print(f"      source text: {snippet}", file=sys.stderr)
    else:
        print("No entries flagged — but always spot-check the output.",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Path to plain-text bibliography (.txt)")
    ap.add_argument("-o", "--output", default=None,
                     help="Path to write the .bib file (default: <input>.bib)")
    ap.add_argument("--split", choices=["auto", "blank", "lines", "numbered"],
                     default="auto",
                     help="How to split the text into entries (default: auto-detect)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    bib_text, records = convert(text, args.split)

    out_path = args.output or re.sub(r"\.txt$", "", args.input, flags=re.I) + ".bib"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(bib_text)

    print_report(records)
    print(f"\nWrote {len(records)} entries to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
