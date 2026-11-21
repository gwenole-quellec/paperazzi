"""On-demand adapters for OpenAlex, Crossref, and imported BibTeX files.

The source layer returns a common record shape understood by ``core``. It does
not write to the database, decide team membership, or merge publications.
All network requests are triggered explicitly by the user interface.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from bibtexparser.bparser import BibTexParser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .core import (
    extract_bibliographic_details,
    normalize_doi,
    normalize_openalex,
    normalize_title,
)

# A connection can fail before a response is returned; a slow OpenAlex query
# may also need more than the default read timeout.
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 45
REQUEST_TIMEOUT = (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)

# Four retries after the initial request, with exponential backoff. All calls
# are GET requests, so retrying them is safe. Respect Retry-After when present.
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)
RETRY_POLICY = Retry(
    total=4,
    connect=4,
    read=4,
    status=4,
    backoff_factor=1.0,
    status_forcelist=RETRYABLE_STATUS_CODES,
    allowed_methods=frozenset({"GET"}),
    respect_retry_after_header=True,
    raise_on_status=False,
)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Paperazzi/0.2 (local research-team bibliography)",
    "Accept": "application/json",
})
SESSION.mount("https://", HTTPAdapter(max_retries=RETRY_POLICY))


def _endpoint_without_query(url: str) -> str:
    """Return a useful endpoint name without query parameters or credentials."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _extract_key_from_url(url: str) -> tuple[str, str]:
    """Strip an OpenAlex API key from a legacy URL before it reaches requests.

    Most Paperazzi calls provide the key through ``openalex_params``. This
    additionally protects accidental direct calls with ``?api_key=...``.
    """
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    api_key = next((value for key, value in query if key.lower() == "api_key"), "")
    filtered_query = [
        (key, value) for key, value in query if key.lower() != "api_key"
    ]
    cleaned = urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(filtered_query),
        parts.fragment,
    ))
    return cleaned, api_key


def api_json(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
) -> dict:
    """Fetch an API JSON document with retries and secret-safe errors.

    OpenAlex authentication is sent through ``Authorization: Bearer`` rather
    than ``api_key`` in the query string. The public function signature stays
    unchanged so existing OpenAlex, Crossref, and test callers still work.

    Transient HTTP failures (429 and 5xx) and connection/read failures are
    automatically retried by the session adapter. Errors never include the
    request query string or the value of the API key.
    """
    safe_url, url_key = _extract_key_from_url(url)
    safe_params = dict(params or {})
    param_key = safe_params.pop("api_key", "")
    api_key = str(param_key or url_key).strip()

    request_headers = dict(headers or {})
    endpoint = _endpoint_without_query(safe_url)
    is_openalex = urlsplit(safe_url).hostname == "api.openalex.org"

    if api_key:
        if not is_openalex:
            raise ValueError("OpenAlex API credentials cannot be sent to another host.")
        request_headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = SESSION.get(
            safe_url,
            params=safe_params,
            headers=request_headers,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        # requests' exception strings may include a URL; never forward them.
        raise RuntimeError(
            f"Request to {endpoint} failed after retries "
            f"({type(exc).__name__}). Try again later."
        ) from None

    status = response.status_code
    if status >= 400:
        if status == 401 and is_openalex:
            detail = "Check that your OpenAlex API key is valid."
        elif status == 429:
            detail = "Rate limit or API budget reached; retry later."
        elif status in RETRYABLE_STATUS_CODES:
            detail = "Temporary server error; retries exhausted."
        else:
            detail = "Check the request or access permissions."
        raise RuntimeError(f"HTTP {status} from {endpoint}. {detail}") from None

    try:
        return response.json()
    except (ValueError, requests.exceptions.RequestException):
        raise RuntimeError(
            f"Invalid JSON response from {endpoint}."
        ) from None


def openalex_params(api_key: str = "", email: str = "") -> dict:
    """Build OpenAlex options; ``api_json`` moves the key into an HTTP header.

    Keep the historical parameter format for compatibility with the existing
    import functions and their tests. The API key is not sent in the URL.
    """
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
        **extract_bibliographic_details("openalex", item),
    }


def openalex_author_profile(author_id: str, api_key: str = "", email: str = "") -> dict:
    """Retrieve an author's canonical name and OpenAlex alternative bylines."""
    match = re.search(r"A\d+", author_id, re.IGNORECASE)
    if not match:
        raise ValueError("Invalid OpenAlex Author ID")
    return api_json(
        "https://api.openalex.org/authors/" + match.group(0).upper(),
        openalex_params(api_key, email),
    )


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
        **extract_bibliographic_details("crossref", item),
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
    parser = BibTexParser(ignore_nonstandard_types=False)
    bibliography = bibtexparser.loads(raw_text, parser=parser)
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
            **extract_bibliographic_details(source, entry),
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
