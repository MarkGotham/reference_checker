# reference_checker

Check on the quality of references in a `.bib` file with respect to:
- completeness of named fields,
- matching items against given APIs.

## Quick start

```
python check_refs.py file_name.bib [options]
```

You supply a BibTeX file to query (`file_name.bib` in the above).

Given that file, this `reference_checker` checks for ... 
- **field completeness.** Does each entry contains a configurable set of required fields?
  - Some are required in all cases: `title`, `author`, `year`.
  - Others Some are type specific (e.g., `doi` for `@article`, `isbn` for `@book`).
- **API match.** Can we find a good match for this entry on major APIs?
  - Default pattern is to query several (up to four) bibliographic APIs in a short-circuit chain.
  - Each bib-API comparison is scored on the closeness of the match.
  - Rules govern early stopping (e.g., DOI match > threshold)

Results are written to a colour-coded PDF report.

## Main Files

| File            | Role                                                                               |
|-----------------|------------------------------------------------------------------------------------|
| `check_refs.py` | CLI entry point (parse arguments, run the async pipeline, call the report builder) |
| `checkers.py`   | API query functions, scoring logic, and field completeness checks                  |
| `report.py`     | ReportLab PDF builder: produces the colour-coded output report                     |

## Usage Options and Configuration

There are CLI options and also user-defined constants (at the top of `checkers.py`).
These mostly cover the thresholds values for flagging and the like.

CrossRef requests that users supply an email address;
this repo has a placeholder value which should be replaced before use.

## Dependencies

```
pip install aiohttp bibtexparser rapidfuzz reportlab tqdm
```

See also `requirements.txt`.

## Licence

[MIT (click here)](./LICENSE).
Mark Gotham, 2026.

## Contribution

Contribution is welcome where there are errors or fixes that apply to the general case.
