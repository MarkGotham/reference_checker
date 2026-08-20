#!/usr/bin/env python3
"""
`checkers.py`

Returns a scored result dict for each .bib entry
based on querying APIs
(some/all of `CrossRef`, `DBLP`, `OpenAlex`, and `Semantic Scholar`).

Score components (each 0 to 1, combined as a weighted average):
- `doi_exact`: DOI found and metadata matches
- `title_sim`: fuzzy similarity match of title between that given and best API hit
- `author_sim`: as for title
- `year_match`: year within 1 of the given year.
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

# Checked for every entry regardless of type.
# TODO confirm no legitimate entry of any kind can omit these.
UNIVERSAL_FIELDS: tuple[str, ...] = ("title", "author", "year")

# Checked only when the entry's entry type matches the key.
# TODO manually made, check all.
ENTRY_TYPE_FIELDS: dict[str, tuple[str, ...]] = {
    "book": ("isbn", "publisher"),
    "article": ("journal", "doi"),
    "inproceedings": ("booktitle", "doi"),  # TODO DOI first, if not ISBN.
    "incollection": ("booktitle", "doi"),  # TODO as above.
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
    Call `log_incomplete_summary` to report problems.
    Do so after the progress bar is closed, otherwise the warning lines
    will print mid-bar and corrupt the output.

    Examples
    --------

    >>> check_required_fields({"ENTRYTYPE": "article", "title": "T", "author": "A", "year": "2020"})
    ['journal', 'doi']
    >>> check_required_fields({"ENTRYTYPE": "book", "title": "T"})
    ['author', 'year', 'isbn', 'publisher']
    >>> check_required_fields({"ENTRYTYPE": "misc", "title": "T", "author": "A", "year": "2020"})
    []
    """
    entry_type = entry.get("ENTRYTYPE", "").lower()
    type_fields = ENTRY_TYPE_FIELDS.get(entry_type, ())
    required = (*UNIVERSAL_FIELDS, *type_fields, *extra)
    return [f for f in required if not str(entry.get(f, "")).strip()]


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
        key = r["entry"].get("ID", "<unknown>")
        missing = ", ".join(r["missing_fields"])
        skipped = "  [API skipped]" if r.get("skipped") else ""
        logger.warning("  - %s: missing %s%s", key, missing, skipped)


# -----------------------------------------------------------------------------

# Helpers

def clean(s: Optional[str]) -> str:
    """
    Strip LaTeX markup and normalise whitespace by removing:
    1. commands (e.g., \\textbf)
    2. braces {}

    Commands are stripped before braces, not after.
    Stripping braces first would turn `\\textbf{Statistics}` into
    `\\textbfStatistics`, and the command regex would then greedily
    consume the whole thing, including the word `Statistics`, as if
    it were all one command name.

    Examples
    --------

    >>> clean(r"{Robust} \\textbf{Statistics}")
    'robust statistics'
    >>> clean("  Multiple   Spaces  ")
    'multiple spaces'
    >>> clean(None)
    ''
    >>> clean("")
    ''
    """
    if not s:
        return ""

    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = re.sub(r"\{([^}]*)}", r"\1", s)

    return " ".join(s.split()).lower()


def title_sim(a: str, b: str) -> float:
    """
    Compare bib file's 'title' field against API data.

    A blank title is a red flag:
    there is nothing to compare, so it is equivalent to the field being absent.
    This always returns 0.0 without calling the fuzzy matcher.
    (Otherwise the calculation is `fuzz.token_sort_ratio("", "")` == 100,
    which would let two missing titles score as a perfect match.)

    Examples
    --------

    >>> title_sim("Deep Learning", "Deep Learning")
    1.0
    >>> round(title_sim("Deep Learning", "Shallow Learning"), 2)
    0.55
    >>> title_sim("", "Deep Learning")
    0.0
    >>> title_sim("", "")
    0.0
    """
    a, b = clean(a), clean(b)
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100


def _normalise_author_name(name: str) -> str:
    """Normalise an author name to 'surname + initials' form.

    Handles three conventions without a comma:
    - "Family, Given" (with comma to disambiguate): e.g. "Smith, John".
    - "Given Family": e.g. "John Smith".
    - "Family Initial": e.g. "Smith J.", (common in citation exports).

    We distinguish "Given Family" and "Family Initial"
    when there is no comma to disambiguate
    by looking at the last token:
    if it is a single letter (optionally followed by a period),
    it cannot be a full surname,
    so it is read as a trailing initial and the first token is the surname.

    This is a heuristic, though are there any a genuine one-letter surnames?
    If it exists, it's certainly extremely rare
    and we have less exacting parts of this code base than that ;)

    Finally, DBLP appends a disambiguation number to a name used more than once
    ("Given Family 0001"),
    so a trailing all-digit token is stripped before parsing
    to prevent it being read as a surname.

    Examples
    --------

    >>> _normalise_author_name("John Smith")
    'smith j'
    >>> _normalise_author_name("Smith, John")
    'smith j'
    >>> _normalise_author_name("J. Smith")
    'smith j'
    >>> _normalise_author_name("Smith, J.")
    'smith j'
    >>> _normalise_author_name("J.R.R. Tolkien")
    'tolkien j r r'
    >>> _normalise_author_name("Smith")
    'smith'
    >>> _normalise_author_name("Michael Jordan 0001")
    'jordan m'
    """
    name = clean(name)
    if not name:
        return ""

    name = re.sub(r"\s+\d+$", "", name)

    if "," in name:
        surname, given = (p.strip() for p in name.split(",", 1))
    else:
        tokens = name.split()
        if len(tokens) == 1:
            return tokens[0]
        last = tokens[-1].rstrip(".")
        if len(last) == 1:
            surname, given = tokens[0], " ".join(tokens[1:])
        else:
            surname, given = tokens[-1], " ".join(tokens[:-1])

    given_tokens = re.split(r"[\s.]+", given)
    initials = " ".join(t[0] for t in given_tokens if t)
    return f"{surname} {initials}".strip()


def _token_lcs_ratio(a: str, b: str) -> float:
    """Longest Common Subsequence ratio on *tokens*.

    Unlike character-level fuzzy matching, a coincidental single-character
    overlap (e.g. the ``j`` in ``smith j`` vs ``jones j``) does not inflate
    the score, only full token matches count.

    Examples
    --------

    >>> _token_lcs_ratio("smith j", "smith j")
    1.0
    >>> _token_lcs_ratio("smith j", "jones b")
    0.0
    >>> round(_token_lcs_ratio("smith j", "smith k"), 2)
    0.5
    >>> round(_token_lcs_ratio("tolkien j r r", "tolkien j"), 2)
    0.5
    """
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    m, n = len(ta), len(tb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ta[i - 1] == tb[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / max(m, n)


def author_sim(bib_authors: str, api_authors: list[str]) -> float:
    """Compare bib 'author' field against a list of strings from API.

    Names are normalised to 'surname + initials' (see
    ``_normalise_author_name``) and then compared **position-wise**:
    each author slot is scored with :func:`_token_lcs_ratio` and the
    per-slot scores are averaged.
    Author order matters, the same names in a different order will
    score 0.0.

    Both ``" and "`` and ``";"`` are treated as author separators,
    covering the two common BibTeX conventions.

    Full name and first-name-initial forms are equivalent::

    Examples
    --------

    >>> author_sim("Smith, John", ["John Smith"])
    1.0
    >>> author_sim("Smith, J.", ["J. Smith"])
    1.0
    >>> author_sim("Smith, John", ["J. Smith"])
    1.0

    Multiple authors in the same order::

    >>> author_sim("Smith, John and Jones, Bob", ["John Smith", "Bob Jones"])
    1.0

    Semicolon-separated authors (non-standard but common in practice)::

    >>> author_sim("Smith, J.; Jones, B.", ["Smith, J.", "Jones, B."])
    1.0

    Wrong order is a fail::

    >>> author_sim("Smith, J. and Jones, B.", ["Bob Jones", "John Smith"])
    0.0

    Completely different authors::

    >>> author_sim("Smith, John", ["Jones, Bob"])
    0.0

    Same surname, different initial (partial match)::

    >>> round(author_sim("Smith, John", ["Smith, Bob"]), 2)
    0.5

    """
    if not bib_authors or not api_authors:
        return 0.0
    bib_list = re.split(r"\s+and\s+|;", bib_authors, flags=re.IGNORECASE)
    bib_norm = [_normalise_author_name(a) for a in bib_list]
    api_norm = [_normalise_author_name(a) for a in api_authors]

    max_len = max(len(bib_norm), len(api_norm))
    scores = []
    for i in range(max_len):
        if i < len(bib_norm) and i < len(api_norm):
            scores.append(_token_lcs_ratio(bib_norm[i], api_norm[i]))
        else:
            scores.append(0.0)  # missing or extra author
    return sum(scores) / len(scores)


def year_match(bib_year: str, api_year) -> float:
    """
    Compare years: all or nothing, matching if at most 1 year apart.

    Examples
    --------

    >>> year_match("2020", "2020")
    1.0
    >>> year_match("2020", "2021")
    1.0
    >>> year_match("2020", "2022")
    0.0
    >>> year_match("2020", None)
    0.0
    >>> year_match("", "2020")
    0.0
    """
    try:
        return 1.0 if abs(int(bib_year) - int(api_year)) <= 1 else 0.0
    except (TypeError, ValueError):
        return 0.0


def journal_sim(bib_venue: str, api_venue: str) -> float:
    """
    Compare venue titles.
    Both sides cleaned so comparisons are consistent regardless of source.

    Examples
    --------

    >>> round(journal_sim("Journal of Machine Learning Research", "J. Mach. Learn. Res."), 2)
    0.57
    >>> journal_sim("Nature", "Nature")
    1.0
    >>> journal_sim("", "Nature")
    0.0
    """
    return fuzz.token_sort_ratio(clean(bib_venue), clean(api_venue)) / 100


def weighted_score(components: dict) -> float:
    """
    Compute a weighted sum of score components using `WEIGHTS`.

    Any key in *components* that is not in `WEIGHTS` is ignored,
    so callers can pass extra bookkeeping keys
    (e.g. `title_missing`)
    alongside the scored ones.

    Examples
    --------

    >>> weighted_score({"doi_exact": 1.0, "title_sim": 1.0, "author_sim": 1.0, "year_match": 1.0, "journal_sim": 1.0})
    1.0
    >>> weighted_score({"doi_exact": 0.0, "title_sim": 0.0, "author_sim": 0.0, "year_match": 0.0, "journal_sim": 0.0})
    0.0
    >>> weighted_score({"doi_exact": 1.0, "title_sim": 0.0, "author_sim": 0.0, "year_match": 0.0, "journal_sim": 0.0})
    0.4
    """
    return sum(WEIGHTS[k] * v for k, v in components.items() if k in WEIGHTS)


async def get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: dict,
) -> Optional[dict]:
    """
    `GET` with simple retry, honouring the `Retry-After` header on 429.

    A 404 (or other non-429 4xx) is a client-side
    "not found" or "bad request"
    and will not change on retry, so it returns `None`
    immediately instead of burning the remaining attempts.
    5xx responses and network errors are treated as transient and retried.
    """
    for attempt in range(3):
        try:
            async with session.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status == 200:
                    return await r.json(content_type=None)
                if r.status == 429:
                    retry_after = int(
                        r.headers.get("Retry-After", RETRY_AFTER_DEFAULT * (attempt + 1))
                    )
                    logger.debug("429 from %s; waiting %ds", url, retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                if 400 <= r.status < 500:
                    logger.debug("%d from %s; not retrying", r.status, url)
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.debug("Request error on attempt %d for %s: %s", attempt, url, exc)
            await asyncio.sleep(2)
    return None


# -----------------------------------------------------------------------------

# Per-API query functions

async def query_crossref(session: aiohttp.ClientSession, entry: dict) -> Optional[dict]:
    """
    Query CrossRef by DOI first, falling back to a title/author search.

    Returns best-match metadata dict or None.
    Author names are always returned as "Family Given" for consistency
    between the DOI and title-search branches.
    """
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
                "source": "CrossRef (DOI)",
                "doi_exact": 1.0,
                "title": " ".join(msg.get("title", [])),
                "authors": [
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
    """
    Return (weighted_score, component_dict) for one API hit.

    A blank bib title is recorded as `title_missing`
    since the biggest component after `doi_exact`
    could not be checked for this entry.
    (The `title_sim` already handles this safely on its own,
    so this is purely a visibility flag for callers.)
    """
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
        "title_missing": not bib_title.strip(),
    }
    return weighted_score(components), components


def _best_of(
    scored: list[tuple[Optional[dict], float, dict]],
) -> tuple[float, Optional[dict], dict]:
    """
    Return the best (score, hit, components) triple,
    skipping entries where the hit is `None`.

    Takes already-scored triples rather than raw hits,
    so a hit that was scored earlier
    (e.g. for a short-circuit check)
    does not need to be scored again here.

    Examples
    --------

    >>> _best_of([(None, 0.0, {}), ({"id": 1}, 0.4, {"a": 1}), ({"id": 2}, 0.9, {"b": 2})])
    (0.9, {'id': 2}, {'b': 2})
    >>> _best_of([(None, 0.0, {})])
    (-1.0, None, {})
    """
    best_score, best_hit, best_components = -1.0, None, {}
    for hit, score, components in scored:
        if hit is None:
            continue
        if score > best_score:
            best_score, best_hit, best_components = score, hit, components
    return best_score, best_hit, best_components


async def score_entry(
    session: aiohttp.ClientSession,
    entry: dict,
    extra: tuple[str, ...] = (),
    skip_if_incomplete: bool = False,
) -> dict:
    """
    Pre-check required fields, then query APIs in order.

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
        If *True* and any required fields are missing,
        skip all API calls and
        return a result with `searched=False` immediately.
        If *False* (default),
        continue with whatever fields exist and
        make the API calls anyway.
        Either way, the missing field list is recorded under `missing_fields`
        on the returned dict, no warning is logged here.
        Call `log_incomplete_summary`
        on the collected results afterwards to report them.

    A blank title is always in `missing_fields`
    (see `UNIVERSAL_FIELDS`),
    and is also visible per-hit as `title_missing` in `components`
    (see `compute_score`),
    since matching without a title to check against is inherently less trustworthy.

    Short-circuit chain (when API calls proceed):
    1. CrossRef: stop if DOI hit scores ≥ `DOI_SHORTCIRCUIT_THRESHOLD`
    2. DBLP: stop if hit scores ≥ `DBLP_SHORTCIRCUIT_THRESHOLD`
    3. OpenAlex + Semantic Scholar in parallel (last resort)

    Note: short-circuiting trades a small chance of a higher score from a
    later API for reduced latency.
    See module-level note.
    """
    missing = check_required_fields(entry, extra)

    if missing and skip_if_incomplete:
        return {
            "entry": entry,
            "score": None,
            "components": {},
            "best_hit": None,
            "searched": False,
            "not_found": False,
            "short_circuit": False,
            "missing_fields": missing,
            "skipped": True,
        }

    # Step 1: CrossRef (best DOI coverage) 
    crossref_hit = await query_crossref(session, entry)
    crossref_score, crossref_components = 0.0, {}
    if crossref_hit is not None:
        crossref_score, crossref_components = compute_score(entry, crossref_hit)
        if crossref_hit.get("doi_exact") and crossref_score >= DOI_SHORTCIRCUIT_THRESHOLD:
            return {
                "entry": entry,
                "score": round(crossref_score, 3),
                "components": crossref_components,
                "best_hit": crossref_hit,
                "searched": True,
                "not_found": False,
                "short_circuit": True,
                "missing_fields": missing,
                "skipped": False,
            }

    # Step 2: DBLP (curated CS metadata, good title matching)
    dblp_hit = await query_dblp(session, entry)
    dblp_score, dblp_components = 0.0, {}
    if dblp_hit is not None:
        dblp_score, dblp_components = compute_score(entry, dblp_hit)
        if dblp_score >= DBLP_SHORTCIRCUIT_THRESHOLD:
            return {
                "entry": entry,
                "score": round(dblp_score, 3),
                "components": dblp_components,
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
    openalex_score, openalex_components = (
        compute_score(entry, openalex_hit) if openalex_hit is not None else (0.0, {})
    )
    ss_score, ss_components = (
        compute_score(entry, ss_hit) if ss_hit is not None else (0.0, {})
    )

    best_score, best_hit, best_components = _best_of([
        (crossref_hit, crossref_score, crossref_components),
        (dblp_hit, dblp_score, dblp_components),
        (openalex_hit, openalex_score, openalex_components),
        (ss_hit, ss_score, ss_components),
    ])

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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
