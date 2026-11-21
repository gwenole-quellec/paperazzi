"""Canonical bibliographic records, source provenance, and manual curation.

Each publication has one canonical row in ``works`` and one or more original
source records in ``variants``. A canonical row is rebuilt from its variants,
while manually edited fields and exclusions remain separate and persistent.

Fuzzy title matches are suggested to users but never merged automatically.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

# Historical source names remain readable so existing databases can be opened.
SOURCE_PRIORITY = {
    "openalex": 0,
    "crossref": 1,
    "bibtex": 2,
    "scholar_bibtex": 3,
    "dblp": 4,
    "orcid": 5,
    "semantic_scholar": 6,
    "openreview": 7,
    "manual": 8,
}
CITATION_PRIORITY = {"openalex": 0, "crossref": 1, "semantic_scholar": 2}

# Includes BibLaTeX extensions alongside classic BibTeX entry types.
BIBTEX_TYPES = (
    "article",
    "book",
    "booklet",
    "conference",
    "inbook",
    "incollection",
    "inproceedings",
    "manual",
    "mastersthesis",
    "misc",
    "online",
    "patent",
    "phdthesis",
    "proceedings",
    "techreport",
    "unpublished",
)

WORK_TYPE_TO_BIBTEX = {
    "journal-article": "article",
    "article": "article",
    "review": "article",
    "proceedings-article": "inproceedings",
    "conference-paper": "inproceedings",
    "conference": "inproceedings",
    "book": "book",
    "edited-book": "book",
    "monograph": "book",
    "book-chapter": "incollection",
    "book-section": "incollection",
    "dissertation": "phdthesis",
    "thesis": "phdthesis",
    "report": "techreport",
    "report-series": "techreport",
    "patent": "patent",
    "preprint": "misc",
    "posted-content": "misc",
}


def infer_bibtex_type(source: str, source_type: str | None) -> str:
    """Convert a provider-specific publication type to a BibTeX entry type.

    Explicit types from BibTeX imports take precedence over index mappings.
    ``book-chapter`` and ``book-section`` become ``incollection``.
    Unknown types are deliberately kept under ``misc`` for manual review.
    """
    kind = str(source_type or "").strip().lower().replace("_", "-")
    if source in {"bibtex", "scholar_bibtex"} and kind in BIBTEX_TYPES:
        return "inproceedings" if kind == "conference" else kind
    return WORK_TYPE_TO_BIBTEX.get(kind, "misc")


def choose_bibtex_type(variants: list[dict]) -> str:
    """Choose the most informative entry type across a work's source records.

    A specific BibTeX entry beats a generic ``misc`` record. Crossref types
    are preferred over other index-inferred types when specificity is equal.
    Manual type overrides are handled separately by :func:`get_works`.
    """
    if not variants:
        return "misc"

    def ranking(variant: dict) -> tuple[int, int, int]:
        source = variant["source"]
        is_generic = (variant.get("bibtex_type") or "misc") == "misc"
        source_rank = 0 if source in {"bibtex", "scholar_bibtex"} else (
            1 if source == "crossref" else 2
        )
        return is_generic, source_rank, SOURCE_PRIORITY.get(source, 99)

    winner = min(variants, key=ranking)
    return winner.get("bibtex_type") or "misc"


def normalize_title(title: str) -> str:
    """Normalize case, accents, markup, and punctuation for title matching."""
    normalized = unicodedata.normalize("NFKD", title or "")
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    normalized = re.sub(r"<[^>]*>", " ", normalized).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def normalize_doi(doi: str | None) -> str:
    """Return a normalized DOI without its URL prefix, or an empty string."""
    if not doi:
        return ""
    value = doi.strip().lower()
    value = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", value)
    return value.rstrip(" .") if value.startswith("10.") else ""


def normalize_openalex(value: str | None) -> str:
    """Extract an OpenAlex Work ID (W...), including from a profile URL."""
    if not value:
        return ""
    match = re.search(r"\bW\d+\b", value, re.IGNORECASE)
    return match.group(0).upper() if match else ""


def int_or_none(value) -> int | None:
    """Parse optional numeric provider fields without failing an entire import."""
    try:
        return int(value) if value not in (None, "") else None
    except (ValueError, TypeError):
        return None


def connect(path: str | Path) -> sqlite3.Connection:
    """Open SQLite, create missing tables, and migrate older databases safely."""
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(database_path), timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS works (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT '',
            year INTEGER,
            doi TEXT,
            venue TEXT,
            authors TEXT,
            work_type TEXT,
            bibtex_type TEXT,
            bibtex_type_override TEXT,
            citations INTEGER NOT NULL DEFAULT 0,
            citation_source TEXT NOT NULL DEFAULT '',
            openalex_id TEXT,
            s2_id TEXT,
            title_override TEXT,
            year_override INTEGER,
            excluded INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS variants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            year INTEGER,
            doi TEXT,
            venue TEXT,
            authors TEXT,
            work_type TEXT,
            bibtex_type TEXT,
            citations INTEGER,
            openalex_id TEXT,
            s2_id TEXT,
            raw_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (source, source_id)
        );
        CREATE INDEX IF NOT EXISTS idx_variants_doi ON variants(doi);
        CREATE INDEX IF NOT EXISTS idx_variants_openalex ON variants(openalex_id);
        CREATE INDEX IF NOT EXISTS idx_variants_work ON variants(work_id);
        """
    )

    # SQLite's ALTER TABLE ADD COLUMN preserves existing works, variants,
    # manual corrections, and team membership relationships.
    migrations = {
        "works": ("bibtex_type", "bibtex_type_override"),
        "variants": ("bibtex_type",),
    }
    for table, expected_columns in migrations.items():
        existing = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        for column in expected_columns:
            if column not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

    missing_types = connection.execute(
        """SELECT id, source, work_type FROM variants
           WHERE bibtex_type IS NULL OR bibtex_type = ''"""
    ).fetchall()
    for row in missing_types:
        connection.execute(
            "UPDATE variants SET bibtex_type=? WHERE id=?",
            (infer_bibtex_type(row["source"], row["work_type"]), row["id"]),
        )

    incomplete_works = connection.execute(
        "SELECT id FROM works WHERE bibtex_type IS NULL OR bibtex_type = ''"
    ).fetchall()
    for row in incomplete_works:
        rebuild(connection, row["id"])

    connection.commit()
    return connection


def _pick(variants: list[dict], field: str):
    """Return the first nonempty field according to source priority."""
    ordered = sorted(variants, key=lambda item: SOURCE_PRIORITY.get(item["source"], 99))
    for variant in ordered:
        if variant.get(field) not in (None, "", []):
            return variant[field]
    return None


def rebuild(connection: sqlite3.Connection, work_id: int) -> None:
    """Recompute canonical metadata without touching manually edited fields."""
    variants = [
        dict(row)
        for row in connection.execute("SELECT * FROM variants WHERE work_id=?", (work_id,))
    ]
    if not variants:
        connection.execute("DELETE FROM works WHERE id=?", (work_id,))
        return

    fields = (
        "title", "year", "doi", "venue", "authors", "work_type", "openalex_id", "s2_id"
    )
    selected = {field: _pick(variants, field) for field in fields}
    entry_type = choose_bibtex_type(variants)
    by_citation_priority = sorted(
        variants, key=lambda item: CITATION_PRIORITY.get(item["source"], 99)
    )
    citation_variant = next(
        (variant for variant in by_citation_priority
         if variant.get("citations") is not None),
        None,
    )
    count = citation_variant["citations"] if citation_variant else 0
    source = citation_variant["source"] if citation_variant else ""

    connection.execute(
        """UPDATE works SET
               title=?, year=?, doi=?, venue=?, authors=?, work_type=?,
               bibtex_type=?, citations=?, citation_source=?, openalex_id=?, s2_id=?
           WHERE id=?""",
        (
            selected["title"] or "", selected["year"], selected["doi"],
            selected["venue"], selected["authors"], selected["work_type"],
            entry_type, count, source, selected["openalex_id"],
            selected["s2_id"], work_id,
        ),
    )


def _matching_work_id(
    connection: sqlite3.Connection, doi: str, openalex_id: str,
    title: str, year: int | None,
) -> int | None:
    """Find safe exact matches by DOI, OpenAlex Work ID, or title and year."""
    if doi:
        row = connection.execute(
            "SELECT work_id FROM variants WHERE doi=? LIMIT 1", (doi,)
        ).fetchone()
        if row:
            return row["work_id"]

    if openalex_id:
        row = connection.execute(
            "SELECT work_id FROM variants WHERE openalex_id=? LIMIT 1", (openalex_id,)
        ).fetchone()
        if row:
            return row["work_id"]

    if year is not None:
        candidates = connection.execute(
            "SELECT id, title, year FROM works WHERE year BETWEEN ? AND ?",
            (year - 1, year + 1),
        ).fetchall()
        normalized_title = normalize_title(title)
        for candidate in candidates:
            if (
                candidate["year"] == year
                and normalize_title(candidate["title"]) == normalized_title
            ):
                return candidate["id"]
    return None


def upsert_records(connection: sqlite3.Connection, records: Iterable[dict]) -> dict:
    """Insert or update source records while preserving canonical work IDs.

    Repeat imports update existing ``(source, source_id)`` variants in place.
    New variants are merged only on safe exact identifiers or title + year.
    """
    totals = {
        "new_works": 0,
        "new_variants": 0,
        "updated_variants": 0,
        "skipped": 0,
    }
    affected_work_ids: set[int] = set()

    for record in records:
        source = str(record.get("source") or "manual").strip()
        source_id = str(record.get("source_id") or "").strip()
        title = str(record.get("title") or "").strip()
        if not source_id or not title:
            totals["skipped"] += 1
            continue

        doi = normalize_doi(record.get("doi"))
        openalex_id = normalize_openalex(record.get("openalex_id"))
        year = int_or_none(record.get("year"))
        citations = int_or_none(record.get("citations"))
        authors = record.get("authors")
        authors = "; ".join(authors) if isinstance(authors, list) else (authors or "")
        entry_type = infer_bibtex_type(source, record.get("work_type"))
        raw_json = json.dumps(
            record.get("raw", record), ensure_ascii=False, default=str
        )

        previous = connection.execute(
            "SELECT id, work_id FROM variants WHERE source=? AND source_id=?",
            (source, source_id),
        ).fetchone()
        fields = (
            title, year, doi, record.get("venue"), authors,
            record.get("work_type"), entry_type, citations,
            openalex_id, record.get("s2_id"), raw_json,
        )

        if previous:
            work_id = previous["work_id"]
            connection.execute(
                """UPDATE variants SET
                       title=?, year=?, doi=?, venue=?, authors=?, work_type=?,
                       bibtex_type=?, citations=?, openalex_id=?, s2_id=?,
                       raw_json=?, updated_at=datetime('now')
                   WHERE id=?""",
                (*fields, previous["id"]),
            )
            totals["updated_variants"] += 1
        else:
            work_id = _matching_work_id(connection, doi, openalex_id, title, year)
            if work_id is None:
                work_id = connection.execute(
                    "INSERT INTO works(title, year) VALUES(?, ?)", (title, year)
                ).lastrowid
                totals["new_works"] += 1

            connection.execute(
                """INSERT INTO variants (
                       work_id, source, source_id, title, year, doi, venue,
                       authors, work_type, bibtex_type, citations,
                       openalex_id, s2_id, raw_json
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (work_id, source, source_id, *fields),
            )
            totals["new_variants"] += 1

        affected_work_ids.add(work_id)

    for work_id in affected_work_ids:
        rebuild(connection, work_id)
    connection.commit()
    return totals


def get_works(connection: sqlite3.Connection, include_excluded: bool = False) -> list[dict]:
    """Return canonical works with manual title, year, and type overrides applied."""
    condition = "" if include_excluded else "WHERE w.excluded=0"
    rows = connection.execute(
        f"""SELECT w.*, COUNT(v.id) AS variant_count,
                   GROUP_CONCAT(DISTINCT v.source) AS sources
            FROM works w
            JOIN variants v ON v.work_id=w.id
            {condition}
            GROUP BY w.id
            ORDER BY COALESCE(w.year_override, w.year) DESC, w.title"""
    ).fetchall()

    works = []
    for row in rows:
        work = dict(row)
        work["title"] = work["title_override"] or work["title"]
        if work["year_override"] is not None:
            work["year"] = work["year_override"]
        work["bibtex_type_inferred"] = work.get("bibtex_type") or "misc"
        work["bibtex_type"] = (
            work.get("bibtex_type_override") or work["bibtex_type_inferred"]
        )
        works.append(work)
    return works


def get_variants(connection: sqlite3.Connection, work_id: int) -> list[dict]:
    """Return the unmodified source records attached to a canonical work."""
    rows = connection.execute(
        "SELECT * FROM variants WHERE work_id=? ORDER BY source, title", (work_id,)
    )
    return [dict(row) for row in rows]


def merge_works(connection: sqlite3.Connection, destination: int, others: list[int]) -> None:
    """Attach selected works' source records to an existing canonical work."""
    if destination in others:
        raise ValueError("The destination publication cannot be among the sources")

    with connection:
        for source in set(others):
            if connection.execute(
                "SELECT 1 FROM works WHERE id=?", (source,)
            ).fetchone() is None:
                continue
            connection.execute(
                "UPDATE variants SET work_id=? WHERE work_id=?", (destination, source)
            )
            connection.execute("DELETE FROM works WHERE id=?", (source,))
        rebuild(connection, destination)


def split_variant(connection: sqlite3.Connection, variant_id: int) -> int:
    """Move one source record into a new canonical work, without deleting it."""
    with connection:
        original = connection.execute(
            "SELECT work_id, title FROM variants WHERE id=?", (variant_id,)
        ).fetchone()
        if original is None:
            raise ValueError("Source record not found")

        count = connection.execute(
            "SELECT COUNT(*) FROM variants WHERE work_id=?", (original["work_id"],)
        ).fetchone()[0]
        if count < 2:
            raise ValueError("This publication has only one source record")

        new_id = connection.execute(
            "INSERT INTO works(title) VALUES(?)", (original["title"],)
        ).lastrowid
        connection.execute(
            "UPDATE variants SET work_id=? WHERE id=?", (new_id, variant_id)
        )
        rebuild(connection, new_id)
        rebuild(connection, original["work_id"])
    return new_id


def set_override(
    connection: sqlite3.Connection, work_id: int,
    title: str | None, year: int | None,
) -> None:
    """Persist optional human corrections without changing the source records."""
    cleaned_title = title.strip() if title and title.strip() else None
    with connection:
        connection.execute(
            "UPDATE works SET title_override=?, year_override=? WHERE id=?",
            (cleaned_title, year, work_id),
        )


def set_bibtex_type_override(
    connection: sqlite3.Connection, work_id: int, entry_type: str | None,
) -> None:
    """Assign or remove a manually selected BibTeX entry type."""
    if entry_type is not None and entry_type not in BIBTEX_TYPES:
        raise ValueError("Unknown BibTeX entry type")
    with connection:
        connection.execute(
            "UPDATE works SET bibtex_type_override=? WHERE id=?",
            (entry_type, work_id),
        )


def set_excluded(connection: sqlite3.Connection, work_id: int, excluded: bool) -> None:
    """Exclude a work from reports without deleting the underlying records."""
    with connection:
        connection.execute(
            "UPDATE works SET excluded=? WHERE id=?", (int(excluded), work_id)
        )


def fuzzy_pairs(
    works: list[dict], threshold: float = 0.78, limit: int = 100,
) -> list[tuple[float, dict, dict]]:
    """Rank possible duplicate pairs by normalized title similarity."""
    suggestions = []
    for index, first in enumerate(works):
        first_title = normalize_title(first["title"])
        if len(first_title) < 8:
            continue
        for second in works[index + 1:]:
            if (
                first["year"] and second["year"]
                and abs(first["year"] - second["year"]) > 3
            ):
                continue
            # Different DOIs can still refer to preprint/published versions.
            score = SequenceMatcher(
                None, first_title, normalize_title(second["title"])
            ).ratio()
            if score >= threshold:
                suggestions.append((score, first, second))
    return sorted(suggestions, key=lambda candidate: candidate[0], reverse=True)[:limit]


def h_index(counts: Iterable[int]) -> int:
    """Compute h-index from one citation count per canonical publication."""
    ordered = sorted((int(value or 0) for value in counts), reverse=True)
    return sum(count >= rank for rank, count in enumerate(ordered, start=1))


def csv_export(works: list[dict]) -> bytes:
    """Export canonical records to UTF-8 CSV with an Excel-friendly BOM."""
    columns = [
        "id", "title", "authors", "year", "venue", "work_type", "bibtex_type",
        "doi", "citations", "citation_source", "openalex_id", "sources", "variant_count",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(works)
    return output.getvalue().encode("utf-8-sig")


def bibtex_export(works: list[dict]) -> bytes:
    """Export canonical works using their effective BibTeX types.

    In particular, book chapters are written as ``@incollection`` with their
    containing book or collection in the ``booktitle`` field.
    """
    def escape_braces(value) -> str:
        return str(value or "").replace("{", r"\{").replace("}", r"\}")

    entries = []
    for work in works:
        last_doi_part = (work.get("doi") or "").split("/")[-1]
        short_key = re.sub(r"[^a-zA-Z0-9_]", "", last_doi_part)[:24]
        key = short_key or f"work{work['id']}"
        entry_type = work.get("bibtex_type") or infer_bibtex_type(
            "crossref", work.get("work_type")
        )
        fields = {
            "title": work["title"],
            "author": (work.get("authors") or "").replace("; ", " and "),
            "year": work.get("year"),
        }
        if entry_type == "article":
            fields["journal"] = work.get("venue")
        elif entry_type in {"inproceedings", "incollection", "inbook"}:
            fields["booktitle"] = work.get("venue")
        elif work.get("venue"):
            fields["note"] = work["venue"]
        fields["doi"] = work.get("doi")

        lines = [f"@{entry_type}{{{key}_w{work['id']},"]
        lines.extend(
            f"  {name} = {{{escape_braces(value)}}},"
            for name, value in fields.items() if value
        )
        entries.append("\n".join(lines) + "\n}")

    return ("\n\n".join(entries) + "\n").encode("utf-8")


def json_export(connection: sqlite3.Connection) -> bytes:
    """Return a provenance-rich JSON reference archive (not a restore format)."""
    works = get_works(connection, include_excluded=True)
    for work in works:
        work["variants"] = get_variants(connection, work["id"])
    return json.dumps(
        {"format": "bibliomap-v1", "works": works},
        ensure_ascii=False, indent=2,
    ).encode("utf-8")
