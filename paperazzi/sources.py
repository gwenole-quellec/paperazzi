"""On-demand adapters for OpenAlex, Crossref, and imported BibTeX files.

The source layer returns a common record shape understood by ``core``. It does
not write to the database, decide team membership, or merge publications.
All network requests are triggered explicitly by the user interface.
"""

from __future__ import annotations

import re

import requests

from .core import normalize_doi, normalize_openalex, normalize_title

TIMEOUT_SECONDS = 25
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Paperazzi/0.3 (local research-team bibliography)",
    "Accept": "application/json",
})


def api_json(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
    """GET a JSON API response, reporting throttling as a readable error."""
    response = SESSION.get(
        url, params=params, headers=headers, timeout=TIMEOUT_SECONDS
    )
    if response.status_code == 429:
        raise RuntimeError(f"API rate limit reached at {response.url}. Try again later.")
    response.raise_for_status()
    return response.json()


def openalex_params(api_key: str = "", email: str = "") -> dict:
    """Include the OpenAlex API key and optional polite-pool email if provided."""
    params = {}
    if api_key:
        params["api_key"] = api_key.strip()
    if email:
        params["mailto"] = email.strip()
    return params


def openalex_authors(query: str, api_key: str = "", orcid: str = "") -> list[dict]:
    """Find author profiles by text search or an exact ORCID filter."""
    params = openalex_params(api_key)
    if orcid:
        identifier = orcid.strip().replace("https://orcid.org/", "")
        params.update({"filter": f"orcid:{identifier}", "per_page": 20})
    else:
        params.update({"search": query, "per_page": 10})
    return api_json("https://api.openalex.org/authors", params).get("results", [])


def from_openalex(item: dict) -> dict:
    """Normalize an OpenAlex Work to Paperazzi's provider-independent shape."""
    primary_location = item.get("primary_location") or {}
    venue = (primary_location.get("source") or {}).get("display_name") or ""
    authors = [
        authorship.get("author", {}).get("display_name")
        or authorship.get("raw_author_name")
        for authorship in item.get("authorships", [])
    ]

    # Crossref's publication type is often more precise for chapters and acts.
    crossref_type = item.get("type_crossref")
    if crossref_type in {"proceedings-article", "journal-article", "book-chapter", "book"}:
        work_type = crossref_type
    else:
        work_type = item.get("type")

    return {
        "source": "openalex",
        "source_id": normalize_openalex(item.get("id")) or str(item.get("id")),
        "title": item.get("display_name") or item.get("title") or "",
        "year": item.get("publication_year"),
        "doi": normalize_doi(item.get("doi")),
        "venue": venue,
        "authors": [author for author in authors if author],
        "work_type": work_type,
        "citations": item.get("cited_by_count"),
        "openalex_id": item.get("id"),
        "raw": item,
    }


def openalex_works(
    author_id: str = "",
    orcid: str = "",
    api_key: str = "",
    email: str = "",
    max_works: int = 2000,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict]:
    """Retrieve one author's works with cursor pagination and inclusive dates.

    OpenAlex Author IDs are preferred for team imports. ORCID support remains
    for compatibility with other clients of this adapter.
    """
    if not (author_id or orcid):
        raise ValueError("Provide an OpenAlex Author ID or ORCID")

    author = re.search(r"A\d+", author_id, re.IGNORECASE)
    if author:
        filter_expression = "author.id:" + author.group(0).upper()
    else:
        orcid_id = orcid.strip().replace("https://orcid.org/", "")
        filter_expression = "authorships.author.orcid:" + orcid_id

    if year_from is not None:
        filter_expression += f",from_publication_date:{int(year_from)}-01-01"
    if year_to is not None:
        filter_expression += f",to_publication_date:{int(year_to)}-12-31"

    records = []
    cursor = "*"
    while cursor and len(records) < max_works:
        params = openalex_params(api_key, email)
        params.update({
            "filter": filter_expression,
            "per_page": min(100, max_works - len(records)),
            "cursor": cursor,
        })
        payload = api_json("https://api.openalex.org/works", params)
        results = payload.get("results", [])
        if not results:
            break
        records.extend(from_openalex(item) for item in results)
        cursor = payload.get("meta", {}).get("next_cursor")
    return records


def openalex_title_search(
    text: str, api_key: str = "", max_results: int = 20,
) -> list[dict]:
    """Find individual OpenAlex works to fill gaps in an author's library."""
    params = openalex_params(api_key)
    params.update({"search": text, "per_page": max_results})
    payload = api_json("https://api.openalex.org/works", params)
    return [from_openalex(item) for item in payload.get("results", [])]


def from_crossref(item: dict) -> dict:
    """Normalize a Crossref Work, including its entry type and container title."""
    date = item.get("published") or item.get("issued") or item.get("created") or {}
    date_parts = date.get("date-parts", [[]])
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    authors = [
        " ".join(filter(None, (author.get("given"), author.get("family"))))
        for author in item.get("author", [])
    ]

    return {
        "source": "crossref",
        "source_id": normalize_doi(item.get("DOI")) or str(item.get("URL", "")),
        "title": (item.get("title") or [""])[0],
        "year": year,
        "doi": item.get("DOI"),
        "venue": (item.get("container-title") or [""])[0],
        "authors": authors,
        "work_type": item.get("type"),
        "citations": item.get("is-referenced-by-count"),
        "raw": item,
    }


def crossref_author_search(
    full_name: str,
    email: str = "",
    rows: int = 300,
    family_name: str = "",
) -> list[dict]:
    """Search Crossref by author, retaining likely surname/initial matches.

    Crossref's ``query.author`` search is fuzzy and can include namesakes.
    Users must still review candidate records before importing them.
    """
    words = normalize_title(full_name).split()
    if not words:
        return []

    last_name = words[-1]
    expected_family = normalize_title(family_name) if family_name else last_name
    first_initial = words[0][0]
    params = {
        "query.author": full_name,
        "rows": min(rows, 1000),
        "select": (
            "DOI,title,author,container-title,published,issued,created,"
            "type,is-referenced-by-count"
        ),
    }
    if email:
        params["mailto"] = email

    payload = api_json("https://api.crossref.org/works", params)
    results = []
    for item in payload.get("message", {}).get("items", []):
        for author in item.get("author", []):
            family = normalize_title(author.get("family", ""))
            given = normalize_title(author.get("given", ""))
            if family in {expected_family, last_name} and (
                not given or given[0] == first_initial
            ):
                record = from_crossref(item)
                if record["title"]:
                    results.append(record)
                break
    return results


def parse_bibtex(
    content: bytes | str,
    source: str = "bibtex",
    namespace: str = "",
) -> list[dict]:
    """Parse BibTeX entries while preserving their explicit ``ENTRYTYPE``.

    A stable per-file namespace ensures that two independent bibliographies
    can reuse the same entry keys without overwriting each other. In particular,
    ``@incollection`` is preserved as a book-chapter type.
    """
    import bibtexparser

    raw_text = (
        content.decode("utf-8-sig", errors="replace")
        if isinstance(content, bytes) else content
    )
    bibliography = bibtexparser.loads(raw_text)
    records = []
    for entry in bibliography.entries:
        if not entry.get("title"):
            continue

        title = entry["title"].replace("{", "").replace("}", "")
        entry_key = entry.get("ID") or normalize_title(title)
        authors = entry.get("author", "")
        records.append({
            "source": source,
            "source_id": f"{namespace}:{entry_key}" if namespace else entry_key,
            "title": title,
            "year": entry.get("year"),
            "doi": entry.get("doi"),
            "venue": entry.get("journal") or entry.get("booktitle") or "",
            "authors": [
                author.strip()
                for author in re.split(r"\s+and\s+", authors)
                if author.strip()
            ],
            "work_type": entry.get("ENTRYTYPE"),
            "citations": None,
            "raw": entry,
        })
    return records


def citing_works(
    openalex_id: str, api_key: str = "", max_results: int = 5,
) -> list[dict]:
    """Retrieve citing works for optional API consumers (not a UI graph)."""
    identifier = normalize_openalex(openalex_id)
    if not identifier:
        return []
    params = openalex_params(api_key)
    params.update({
        "filter": "cites:" + identifier,
        "per_page": max_results,
        "sort": "-cited_by_count",
    })
    payload = api_json("https://api.openalex.org/works", params)
    return [from_openalex(item) for item in payload.get("results", [])]


def crossref_by_doi(doi: str, email: str = "") -> dict:
    """Fetch one Crossref record by its exact DOI."""
    identifier = normalize_doi(doi)
    if not identifier:
        raise ValueError("Invalid DOI")
    params = {"mailto": email} if email else {}
    url = "https://api.crossref.org/works/" + requests.utils.quote(identifier, safe="/")
    payload = api_json(url, params)
    return from_crossref(payload["message"])
