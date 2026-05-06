#!/usr/bin/env python3
"""
`checkers.py`

Returns a scored result dict for each .bib entry
based on querying APIs
(some/all of `CrossRef`, `DBLP`, `OpenAlex`, and `Semantic Scholar`).

Score components (each 0–1, combined as a weighted average):
- `doi_exact`: DOI found and metadata matches
- `title_sim`: fuzzy similarity match of title between that given and best API hit
- `author_sim`: as for title
- `year_match`: Year within ±1
- `journal_sim`: Venue (e.g., journal) name similarity

Short-circuit chain:
1. CrossRef: stop if strong DOI match (score ≥ DOI_SHORTCIRCUIT_THRESHOLD)
2. DBLP: stop if strong title/author match (score ≥ DBLP_SHORTCIRCUIT_THRESHOLD)
3. OpenAlex + Semantic Scholar run in parallel (fallback)

Note on short-circuiting:
Given the early exits, we may miss a higher-scoring hit from a later API.
We trade this for reduced latency and fewer API calls.
You, dear user, can adjust the thresholds if completeness matters more than speed.
"""

import asyncio
import logging
import re
from typing import Optional

import aiohttp
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


__author__ = "Mark Gotham"


# -----------------------------------------------------------------------------

# Configuration (only used in this file)

# Email shown in the User-Agent header.  CrossRef's polite pool requires this.
# TODO replace placeholder with your info
CONTACT_EMAIL = "your@email.here"

HEADERS = {"User-Agent": f"BibRefChecker/1.0 (mailto:{CONTACT_EMAIL})"}

WEIGHTS = {
    "doi_exact":   0.40,
    "title_sim":   0.30,
    "author_sim":  0.15,
    "year_match":  0.10,
    "journal_sim": 0.05,
}

# Default retry delay (seconds) when a 429 response has no Retry-After header.
RETRY_AFTER_DEFAULT = 5

# Number of results to request from each API per query.
MAX_HITS_PER_QUERY = 1

# Score thresholds for short-circuiting the API chain.
# Note: early exit means a later API cannot produce a higher score.
DOI_SHORTCIRCUIT_THRESHOLD  = 0.85   # CrossRef DOI hit: skip all remaining APIs
DBLP_SHORTCIRCUIT_THRESHOLD = 0.80   # DBLP hit: skip OpenAlex + Semantic Scholar

# Required-field pre-check 

# Checked for every entry regardless of type. TODO right? No legit cases can omit these?
UNIVERSAL_FIELDS: tuple[str, ...] = ("title", "author", "year")

# Checked only when the entry's ENTRY_TYPE matches the key. TODO manually made, check all
ENTRY_TYPE_FIELDS: dict[str, tuple[str, ...]] = {
    "book": ("isbn", "publisher"),
    "article": ("journal",),
    "inproceedings": ("booktitle",),
    "incollection": ("booktitle",),
    "phdthesis": ("school",),
}


# -----------------------------------------------------------------------------

def check_required_fields(
    entry: dict,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """
    Return a list of field names that are absent or blank in *entry*.

    This function checks
    the `UNIVERSAL_FIELDS` (title, author, year in every case),
    the fields listed in `ENTRY_TYPE_FIELDS` for each given entry's type,
    and any caller-supplied at the CLI's `--required` option.

    This function collects missing field names.
    It does not log or print anything.
    Call `log_incomplete_summary` to report problems, and do so
    after the progress bar is closed otherwise the warning lines will print
    mid-bar and corrupt the output.
    """
    entry_type = entry.get("ENTRY_TYPE", "").lower()
    type_fields = ENTRY_TYPE_FIELDS.get(entry_type, ())
    required = (*UNIVERSAL_FIELDS, *type_fields, *extra)
    return [f for f in required if not entry.get(f, "").strip()]


def log_incomplete_summary(results: list[dict]) -> None:
    """
    Log a grouped summary of all incomplete entries.

    Call this *after* the processing loop so warnings don't interleave with
    progress output.  Does nothing if every entry is complete.
    """
    incomplete = [r for r in results if r.get("missing_fields")]
    if not incomplete:
        return
    logger.warning("INCOMPLETE ENTRIES (%d of %d)", len(incomplete), len(results))
    for r in incomplete:
        key     = r["entry"].get("ID", "<unknown>")
        missing = ", ".join(r["missing_fields"])
        skipped = "  [API skipped]" if r.get("skipped") else ""
        logger.warning("  • %s: missing %s%s", key, missing, skipped)


# -----------------------------------------------------------------------------

# Helpers

def clean(s: Optional[str]) -> str:
    """
    Strip LaTeX markup and normalise whitespace by removing:
    1. braces {}
    2. commands (e.g., \textbf)
    """
    if not s:
        return ""
    s = re.sub(r"\{([^}]*)}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    return " ".join(s.split()).lower()


def title_sim(a: str, b: str) -> float:
    """Compare bib file's 'title' field against API data."""
    return fuzz.token_sort_ratio(clean(a), clean(b)) / 100


def author_sim(bib_authors: str, api_authors: list[str]) -> float:
    """Compare bib 'author' field against a list of strings from API."""
    if not bib_authors or not api_authors:
        return 0.0
    bib = clean(bib_authors)
    api = " ".join(clean(a) for a in api_authors)
    return fuzz.token_set_ratio(bib, api) / 100


def year_match(bib_year: str, api_year) -> float:
    """Compare years (max 1 year apart, nall or nothing)."""
    try:
        return 1.0 if abs(int(bib_year) - int(api_year)) <= 1 else 0.0
    except (TypeError, ValueError):
        return 0.0


def journal_sim(bib_venue: str, api_venue: str) -> float:
    """
    Compare venue titles.
    Both sides cleaned so comparisons are consistent regardless of source.
    """
    return fuzz.token_sort_ratio(clean(bib_venue), clean(api_venue)) / 100


def weighted_score(components: dict) -> float:
    """Compute weighted score."""
    return sum(WEIGHTS[k] * v for k, v in components.items() if k in WEIGHTS)


async def get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: dict,
) -> Optional[dict]:
    """`GET` with simple retry on 429, honouring the `Retry-After` header."""
    for attempt in range(3):
        try:
            async with session.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status == 429:
                    retry_after = int(
                        r.headers.get("Retry-After", RETRY_AFTER_DEFAULT * (attempt + 1))
                    )
                    logger.debug("429 from %s; waiting %ds", url, retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                if r.status == 200:
                    return await r.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.debug("Request error on attempt %d for %s: %s", attempt, url, exc)
            await asyncio.sleep(2)
    return None


# -----------------------------------------------------------------------------

# Per-API query functions

async def query_crossref(session: aiohttp.ClientSession, entry: dict) -> Optional[dict]:
    """Returns best-match metadata dict or None."""
    doi = entry.get("doi", "").strip()

    # 1. DOI direct lookup (highest confidence)
    if doi:
        data = await get_json(session, f"https://api.crossref.org/works/{doi}", {})
        if data and data.get("status") == "ok":
            msg = data["message"]
            date_parts = (
                (msg.get("published-print") or msg.get("published-online") or {})
                .get("date-parts", [[None]])[0]
            )
            year_raw = date_parts[0] if date_parts else None
            return {
                "source":    "CrossRef (DOI)",
                "doi_exact": 1.0,
                "title":     " ".join(msg.get("title", [])),
                "authors":   [
                    f"{a.get('family', '')} {a.get('given', '')}".strip()
                    for a in msg.get("author", [])
                ],
                "year": str(year_raw) if year_raw is not None else "",
                "venue": clean(msg.get("container-title", [""])[0]),
            }

    # 2. Title search fallback
    title = entry.get("title", "")
    if not title:
        return None
    data = await get_json(session, "https://api.crossref.org/works", {
        "query.title":  clean(title),
        "query.author": clean(entry.get("author", "")),
        "rows": MAX_HITS_PER_QUERY,
        "select": "DOI,title,author,published-print,published-online,container-title",
    })
    if not data:
        return None
    items = data.get("message", {}).get("items", [])
    if not items:
        return None
    msg = items[0]
    date_parts = (
        (msg.get("published-print") or msg.get("published-online") or {})
        .get("date-parts", [[None]])[0]
    )
    year_raw = date_parts[0] if date_parts else None
    return {
        "source": "CrossRef (search)",
        "doi_exact": 0.0,
        "title": " ".join(msg.get("title", [])),
        "authors": [
            f"{a.get('family', '')} {a.get('given', '')}".strip()
            for a in msg.get("author", [])
        ],
        "year": str(year_raw) if year_raw is not None else "",
        "venue": clean(msg.get("container-title", [""])[0]),
    }


async def query_dblp(session: aiohttp.ClientSession, entry: dict) -> Optional[dict]:
    """Query DBLP publication search API.

    DBLP covers CS literature very well and returns clean, curated metadata.
    We search by title (+ first-author surname if available) using DBLP's
    CompleteSearch engine, which is tolerant of minor variations.

    API docs: https://dblp.org/faq/How+to+use+the+dblp+search+API.html
    Endpoint: https://dblp.org/search/publ/api
    """
    title = entry.get("title", "")
    if not title:
        return None

    # Build query: title tokens + first-author surname if available.
    query_parts = [clean(title)]
    author_field = entry.get("author", "")
    if author_field:
        first_author = re.split(r"\s+and\s+", author_field, flags=re.IGNORECASE)[0]
        surname = first_author.split(",")[0].strip()
        if surname:
            query_parts.append(clean(surname))
    query = " ".join(query_parts)

    data = await get_json(session, "https://dblp.org/search/publ/api", {
        "q": query,
        "format": "json",
        "h": MAX_HITS_PER_QUERY,
        "c": 0,
    })
    if not data:
        return None

    hits = (data.get("result") or {}).get("hits") or {}
    hit_list = hits.get("hit") or []
    if not hit_list:
        return None

    info = hit_list[0].get("info", {})
    if not info:
        return None

    raw_authors = info.get("authors", {}).get("author", [])
    if isinstance(raw_authors, str):
        authors = [raw_authors]
    elif isinstance(raw_authors, dict):
        authors = [raw_authors.get("text", "")]
    else:
        authors = [
            a.get("text", a) if isinstance(a, dict) else a
            for a in raw_authors
        ]

    venue = info.get("venue") or info.get("journal") or info.get("booktitle") or ""

    return {
        "source": "DBLP",
        "doi_exact": 0.0,
        "title": info.get("title", ""),
        "authors": authors,
        "year": str(info.get("year", "")),
        "venue": clean(venue),
    }


async def query_openalex(session: aiohttp.ClientSession, entry: dict) -> Optional[dict]:
    doi = entry.get("doi", "").strip()
    title = entry.get("title", "")

    params: dict = {"select": "title,authorships,publication_year,primary_location"}
    if doi:
        params["filter"] = f"doi:{doi}"
    elif title:
        params["search"] = clean(title)
        params["per-page"] = str(MAX_HITS_PER_QUERY)
    else:
        return None

    data = await get_json(session, "https://api.openalex.org/works", params)
    if not data:
        return None
    results = data.get("results", [])
    if not results:
        return None
    w = results[0]
    authors = [a["author"].get("display_name", "") for a in w.get("authorships", [])]
    source = ((w.get("primary_location") or {}).get("source") or {})
    return {
        "source": "OpenAlex",
        "doi_exact": 0.0,
        "title": w.get("title", ""),
        "authors": authors,
        "year": str(w.get("publication_year", "")),
        "venue": clean(source.get("display_name", "")),
    }


async def query_semantic_scholar(
    session: aiohttp.ClientSession, entry: dict
) -> Optional[dict]:
    doi = entry.get("doi", "").strip()
    title = entry.get("title", "")
    fields = "title,authors,year,venue,externalIds"

    if doi:
        data = await get_json(
            session,
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            {"fields": fields},
        )
        if data and "title" in data:
            return {
                "source": "Semantic Scholar (DOI)",
                "doi_exact": 0.0,
                "title": data.get("title", ""),
                "authors": [a.get("name", "") for a in data.get("authors", [])],
                "year": str(data.get("year", "")),
                "venue": clean(data.get("venue", "")),
            }

    if not title:
        return None
    data = await get_json(
        session,
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": clean(title), "fields": fields, "limit": MAX_HITS_PER_QUERY},
    )
    if not data:
        return None
    items = data.get("data", [])
    if not items:
        return None
    p = items[0]
    return {
        "source": "Semantic Scholar (search)",
        "doi_exact": 0.0,
        "title": p.get("title", ""),
        "authors": [a.get("name", "") for a in p.get("authors", [])],
        "year": str(p.get("year", "")),
        "venue": clean(p.get("venue", "")),
    }


# -----------------------------------------------------------------------------

# Scoring

def compute_score(entry: dict, hit: dict) -> tuple[float, dict]:
    """Return (weighted_score, component_dict) for one API hit."""
    bib_title = entry.get("title", "")
    bib_authors = entry.get("author", "")
    bib_year = entry.get("year", "")
    bib_venue = entry.get("journal") or entry.get("booktitle", "")

    components = {
        "doi_exact": hit.get("doi_exact", 0.0),
        "title_sim": title_sim(bib_title, hit.get("title", "")),
        "author_sim": author_sim(bib_authors, hit.get("authors", [])),
        "year_match": year_match(bib_year, hit.get("year")),
        "journal_sim": journal_sim(bib_venue, hit.get("venue", "")),
    }
    return weighted_score(components), components


def _pick_best(
    entry: dict,
    hits: list[Optional[dict]],
) -> tuple[float, Optional[dict], dict]:
    """Return (best_score, best_hit, best_components) across all hits."""
    best_score, best_hit, best_components = -1.0, None, {}
    for hit in hits:
        if hit is None:
            continue
        score, components = compute_score(entry, hit)
        if score > best_score:
            best_score, best_hit, best_components = score, hit, components
    return best_score, best_hit, best_components


async def score_entry(
    session: aiohttp.ClientSession,
    entry: dict,
    extra: tuple[str, ...] = (),
    skip_if_incomplete: bool = False,
) -> dict:
    """Pre-check required fields, then query APIs in order.

    Parameters
    ----------
    session:
        A shared `aiohttp.ClientSession`.
        Create one session per batch and pass it here so connection pools are reused.
    entry:
        Parsed .bib entry dict.
    extra:
        Additional fields to require beyond the automatic universal and entry-type-conditional set.
        Passed straight through to `check_required_fields`.
        Typically populated from the CLI `--required` option (e.g. `("doi",)`).
    skip_if_incomplete:
        If *True* and any required fields are missing, skip all API calls and
        return a result with `searched=False` immediately.
        The missing field list is recorded under `missing_fields`.
        If *False* (default), warn but continue with whatever fields exist.

    Short-circuit chain (when API calls proceed):
        1. CrossRef  → stop if DOI hit scores ≥ `DOI_SHORTCIRCUIT_THRESHOLD`
        2. DBLP      → stop if hit scores ≥ `DBLP_SHORTCIRCUIT_THRESHOLD`
        3. OpenAlex + Semantic Scholar in parallel (last resort)

    Note: short-circuiting trades a small chance of a higher score from a
    later API for reduced latency.  See module-level note.
    """
    missing = check_required_fields(entry, extra)

    if missing and skip_if_incomplete:
        return {
            "entry": entry,
            "score": 0.0,
            "components": {},
            "best_hit": None,
            "searched": False,  # API calls were not made
            "not_found": False,
            "short_circuit": False,
            "missing_fields": missing,
            "skipped": True,
        }

    # Step 1: CrossRef (best DOI coverage) 
    crossref_hit = await query_crossref(session, entry)
    if crossref_hit is not None:
        score, components = compute_score(entry, crossref_hit)
        if crossref_hit.get("doi_exact") and score >= DOI_SHORTCIRCUIT_THRESHOLD:
            return {
                "entry": entry,
                "score": round(score, 3),
                "components": components,
                "best_hit": crossref_hit,
                "searched": True,
                "not_found": False,
                "short_circuit": True,
                "missing_fields": missing,
                "skipped": False,
            }

    # Step 2: DBLP (curated CS metadata, good title matching)
    dblp_hit = await query_dblp(session, entry)
    if dblp_hit is not None:
        score, components = compute_score(entry, dblp_hit)
        if score >= DBLP_SHORTCIRCUIT_THRESHOLD:
            return {
                "entry": entry,
                "score": round(score, 3),
                "components": components,
                "best_hit": dblp_hit,
                "searched": True,
                "not_found": False,
                "short_circuit": True,
                "missing_fields": missing,
                "skipped": False,
            }

    # Step 3: OpenAlex + Semantic Scholar in parallel (fallback)
    openalex_hit, ss_hit = await asyncio.gather(
        query_openalex(session, entry),
        query_semantic_scholar(session, entry),
    )

    best_score, best_hit, best_components = _pick_best(
        entry, [crossref_hit, dblp_hit, openalex_hit, ss_hit]
    )

    return {
        "entry": entry,
        "score": round(best_score, 3) if best_hit else None,
        "components": best_components,
        "best_hit": best_hit,
        "searched": True,
        "not_found": best_hit is None,
        "short_circuit": False,
        "missing_fields": missing,
        "skipped": False,
    }
