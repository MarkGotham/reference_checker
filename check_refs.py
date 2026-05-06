#!/usr/bin/env python3
"""
`check_refs.py`

Check on the quality of .bib files.
1. Checks required fields (e.g. author, year)
2. Makes API calls to CrossRef, DBLP, OpenAlex, and Semantic Scholar
3. Scores each reference and outputs a colour-coded PDF report.

Usage:
    python check_refs.py file_name.bib [options]

Options:
    --amber FLOAT           Score threshold below which a ref is flagged amber (default 0.7)
    --red   FLOAT           Score threshold below which a ref is flagged red (default 0.4)
    --out   PATH            Output PDF path  (default: <bibfile>_ref_qual_report.pdf)
    --concurrency INT       Max simultaneous API requests  (default: 5)
    --required FIELD ...    Extra fields to require on top of the automatic set.
                            For all cases: title, author, year.
                            Depending on the entry type:
                            @article requires `journal`, @book requires `isbn` and `publisher`
                            etc.
                            Default is none.
    --skip-incomplete       Skip API calls for entries with missing required fields
                            and flag them red in the report
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import aiohttp
import bibtexparser
from tqdm import tqdm

from checkers import ENTRY_TYPE_FIELDS, UNIVERSAL_FIELDS, log_incomplete_summary, score_entry, check_required_fields
from report import build_report

logger = logging.getLogger(__name__)

__author__ = "Mark Gotham"


# -----------------------------------------------------------------------------

# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("bibfile", help="Path to the .bib file")
    p.add_argument("--amber",       type=float, default=0.7,
                   help="Score below this → amber  (default 0.7)")
    p.add_argument("--red",         type=float, default=0.4,
                   help="Score below this → red    (default 0.4)")
    p.add_argument("--out",         type=str,   default=None,
                   help="Output PDF path  (default: <bibfile>_ref_qual_report.pdf)")
    p.add_argument("--concurrency", type=int,   default=5,
                   help="Max simultaneous API requests (default 5)")
    p.add_argument("--required",    type=str,   nargs="*",
                   default=[],
                   metavar="FIELD",
                   help=(
                       "Extra fields to require on top of the automatic set "
                       "(universal: %(universal)s; plus per type: %(per_type)s). "
                       "Default: none."
                       % {
                           "universal": ", ".join(UNIVERSAL_FIELDS),
                           "per_type":  ", ".join(
                               f"@{t}: {', '.join(fs)}"
                               for t, fs in ENTRY_TYPE_FIELDS.items()
                           ),
                       }
                   ))
    p.add_argument("--skip-incomplete", action="store_true", default=False,
                   help="Skip API calls for incomplete entries and flag them red")
    return p.parse_args()


# -----------------------------------------------------------------------------

# Entry point

async def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,  # keep logging and tqdm (stderr) on separate streams
    )

    args = parse_args()
    bib_path = Path(args.bibfile)
    if not bib_path.exists():
        sys.exit(f"File not found: {bib_path}")

    out_path = (
        Path(args.out)
        if args.out
        else bib_path.with_name(bib_path.stem + "_ref_qual_report.pdf")
    )
    extra = tuple(args.required or [])

    # Parse .bib
    with open(bib_path, encoding="utf-8") as fh:
        library = bibtexparser.load(fh)
    bib_entries = library.entries
    print(f"Loaded {len(bib_entries)} references from {bib_path.name}")
    if args.skip_incomplete:
        extra_note = f", plus extra: {', '.join(extra)}" if extra else ""
        print(
            f"  Incomplete entries (always checked: {', '.join(UNIVERSAL_FIELDS)}; "
            f"per-type conditional fields apply{extra_note}) "
            "will be flagged red without API calls."
        )

    # Score all entries (bounded concurrency) with a progress bar.
    # tqdm auto-suppresses the bar when stderr is not a TTY (e.g. piped/logged).
    sem     = asyncio.Semaphore(args.concurrency)
    results = []

    async def bounded(session: aiohttp.ClientSession, entry: dict) -> None:
        async with sem:
            try:
                result = await score_entry(
                    session,
                    entry,
                    extra=extra,
                    skip_if_incomplete=args.skip_incomplete,
                )
            except Exception as exc:
                key = entry.get("ID", "<unknown>")
                logger.warning("Unhandled error scoring %s: %s", key, exc)

                # Calculate missing fields to ensure they are reported
                missing_fields = check_required_fields(entry, extra=extra)

                result = {
                    "entry": entry,
                    "score": None,
                    "components": {},
                    "best_hit": None,
                    "searched": False,
                    "not_found": True,
                    "short_circuit": False,
                    "missing_fields": missing_fields,
                    "skipped": False,
                    "error": str(exc),
                }
        results.append(result)
        bar.update(1)

    bar = tqdm(total=len(bib_entries), unit="ref", dynamic_ncols=True)
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[bounded(session, e) for e in bib_entries])
    bar.close()

    # Summarise short-circuits (derived from results — no shared mutable counter).
    short_circuits = sum(1 for r in results if r.get("short_circuit"))
    bar.set_postfix(shortcircuit=short_circuits)

    # Log incomplete-entry warnings after the bar is closed so they don't interleave.
    log_incomplete_summary(results)

    print(f"  {short_circuits}/{len(bib_entries)} resolved via short-circuit "
          "(skipped OpenAlex + S2)")

    # Build report
    print("Writing report ...")
    build_report(results, out_path, amber=args.amber, red=args.red)
    print(f"Done → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
