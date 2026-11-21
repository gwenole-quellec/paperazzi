"""Research-team membership, authorship attribution, and report exports.

The data model separates three concepts:

* ``works``: one canonical publication, regardless of how many colleagues
  authored it or how many metadata sources describe it;
* ``variant_members``: author associations inferred from individual sources;
* ``member_work_overrides``: explicit corrections that survive reimports.

Reporting-year boundaries only affect views and OpenAlex search limits. They
never delete existing works or their full authorship provenance.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .core import (
    connect,
    get_works,
    get_variants,
    normalize_title,
    upsert_records,
)


def normalized_author_id(value: str) -> str:
    """Return an OpenAlex Author ID, accepting both short IDs and profile URLs."""
    text = str(value or "").strip()
    match = re.search(r"(?:^|[/\s])A\d+(?:/?$)", text, re.IGNORECASE)
    if match:
        identifier = re.search(r"A\d+", match.group(0), re.IGNORECASE)
        if identifier:
            return identifier.group(0).upper()
    if re.fullmatch(r"A\d+", text, re.IGNORECASE):
        return text.upper()
    raise ValueError(
        f"Invalid OpenAlex Author ID: {value!r}. Expected A123456789 "
        "or an OpenAlex author profile URL."
    )


def parse_author_ids(value: str | Iterable[str]) -> list[str]:
    """Parse and deduplicate a list of author IDs without changing their order."""
    raw_values = (
        re.split(r"[,;\s]+", value.strip()) if isinstance(value, str) else list(value)
    )
    identifiers = []
    for raw in raw_values:
        if not str(raw).strip():
            continue
        identifier = normalized_author_id(str(raw))
        if identifier not in identifiers:
            identifiers.append(identifier)
    return identifiers



def normalize_author_name(name: str) -> str:
    """Canonicalize a byline, including BibTeX's ``Family, Given`` spelling.

    Only exact normalized aliases are matched, never guessed initials or
    partial surnames. This intentionally favors precision over recall.
    """
    text = str(name or "").strip()
    # Simple TeX accent commands are common in Google Scholar BibTeX exports.
    # Decode them before removing braces and normalizing Unicode accents.
    accent_marks = {"'": "\u0301", "`": "\u0300", '^': "\u0302",
                    '"': "\u0308", '~': "\u0303", 'c': "\u0327"}
    for symbol, combining in accent_marks.items():
        text = re.sub(
            r"\\" + re.escape(symbol) + r"\s*\{?([A-Za-z])\}?",
            lambda match: match.group(1) + combining,
            text,
        )
    text = text.replace("{", "").replace("}", "")
    if "," in text:
        family, given = text.split(",", 1)
        text = f"{given.strip()} {family.strip()}"
    return normalize_title(text)


def cached_alias_index(connection) -> dict[str, int]:
    """Map unambiguous OpenAlex bylines and roster names to team members.

    OpenAlex alternative bylines can be noisy and shared by namesakes.
    Duplicated normalized aliases are deliberately excluded from automatic
    assignment until a person explicitly reviews the publication.
    """
    names = defaultdict(set)
    members = list_members(connection)
    for member in members:
        names[normalize_author_name(member["name"])].add(member["id"])
        for author_id in member["openalex_ids"]:
            profile = connection.execute(
                "SELECT aliases_json FROM author_profiles WHERE openalex_id=?",
                (author_id,),
            ).fetchone()
            if profile:
                for alias in json.loads(profile["aliases_json"]):
                    names[normalize_author_name(alias)].add(member["id"])
    return {alias: next(iter(owners)) for alias, owners in names.items()
            if alias and len(owners) == 1}


def save_author_profile(connection, author_id: str, profile: dict) -> None:
    """Cache OpenAlex name alternatives and reindex existing BibTeX records.

    The cache belongs to the OpenAlex ID rather than a member, preventing
    stale aliases from being reused after a member's IDs are changed.
    """
    identifier = normalized_author_id(author_id)
    names = [profile.get("display_name"),
             *(profile.get("display_name_alternatives") or [])]
    cleaned = sorted({str(name).strip() for name in names if name})
    with connection:
        connection.execute(
            """INSERT INTO author_profiles (openalex_id, aliases_json)
               VALUES (?,?) ON CONFLICT(openalex_id)
               DO UPDATE SET aliases_json=excluded.aliases_json""",
            (identifier, json.dumps(cleaned, ensure_ascii=False)),
        )
        reindex_bibtex_members(connection)


def missing_author_profiles(connection, members: list[dict] | None = None) -> list[str]:
    """List configured author IDs with no cached OpenAlex name alternatives."""
    registered = members if members is not None else list_members(connection)
    identifiers = {identifier for member in registered for identifier in member["openalex_ids"]}
    return sorted(identifier for identifier in identifiers if connection.execute(
        "SELECT 1 FROM author_profiles WHERE openalex_id=?", (identifier,)
    ).fetchone() is None)


def reindex_bibtex_members(connection) -> None:
    """Rebuild *only* inferred BibTeX name evidence; keep manual decisions.

    Existing BibTeX variants are rechecked whenever a profile, member or
    author identifier changes. Ambiguous aliases do not create links.
    """
    index = cached_alias_index(connection)
    connection.execute("DELETE FROM variant_members WHERE evidence='bibtex_alias'")
    for variant in connection.execute(
        "SELECT id, authors FROM variants WHERE source IN ('bibtex','scholar_bibtex')"
    ):
        for byline in (variant["authors"] or "").split("; "):
            member_id = index.get(normalize_author_name(byline))
            if member_id is not None:
                connection.execute(
                    """INSERT OR IGNORE INTO variant_members
                       (variant_id, member_id, evidence) VALUES (?,?,'bibtex_alias')""",
                    (variant["id"], member_id),
                )

def connect_team(path: str | Path):
    """Open the local database and create or migrate its team-related tables."""
    connection = connect(path)
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS author_ids (
            openalex_id TEXT PRIMARY KEY,
            member_id INTEGER NOT NULL REFERENCES members(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS variant_members (
            variant_id INTEGER NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
            member_id INTEGER NOT NULL REFERENCES members(id) ON DELETE CASCADE,
            evidence TEXT NOT NULL,
            PRIMARY KEY(variant_id, member_id, evidence)
        );
        CREATE TABLE IF NOT EXISTS member_work_overrides (
            work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
            member_id INTEGER NOT NULL REFERENCES members(id) ON DELETE CASCADE,
            decision TEXT NOT NULL CHECK(decision IN ('include', 'exclude')),
            PRIMARY KEY(work_id, member_id)
        );
        CREATE TABLE IF NOT EXISTS author_profiles (
            openalex_id TEXT PRIMARY KEY,
            aliases_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_author_ids_member
            ON author_ids(member_id);
        CREATE INDEX IF NOT EXISTS idx_variant_members_member
            ON variant_members(member_id);
        CREATE INDEX IF NOT EXISTS idx_variant_members_variant
            ON variant_members(variant_id);
        """
    )

    # Migration from databases created before per-member year bounds existed.
    known_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(members)")
    }
    with connection:
        for field in ("publication_year_from", "publication_year_to"):
            if field not in known_columns:
                connection.execute(f"ALTER TABLE members ADD COLUMN {field} INTEGER")
    return connection


def _validated_years(year_from, year_to) -> tuple[int | None, int | None]:
    """Validate optional inclusive year limits; ``None`` means no limit."""
    start, end = (
        None if value is None or value == "" else int(value)
        for value in (year_from, year_to)
    )
    if (
        (start is not None and not 1000 <= start <= 2200)
        or (end is not None and not 1000 <= end <= 2200)
    ):
        raise ValueError("Publication years must be between 1000 and 2200")
    if start is not None and end is not None and start > end:
        raise ValueError("Start year cannot be later than end year")
    return start, end


def member_import_bounds(
    member: dict,
    global_from: int | None = None,
    global_to: int | None = None,
) -> tuple[int | None, int | None, bool]:
    """Intersect individual and optional global year ranges for API searches.

    Return ``(lower, upper, empty)``. ``empty=True`` means the two ranges do
    not overlap; no search should be attempted for that researcher.
    """
    lower_candidates = [
        year for year in (member["publication_year_from"], global_from)
        if year is not None
    ]
    upper_candidates = [
        year for year in (member["publication_year_to"], global_to)
        if year is not None
    ]
    lower = max(lower_candidates) if lower_candidates else None
    upper = min(upper_candidates) if upper_candidates else None
    empty = lower is not None and upper is not None and lower > upper
    return lower, upper, empty


def scope_team_works(
    works: list[dict],
    members: list[dict],
    selected_ids: Iterable[int],
    global_from: int | None = None,
    global_to: int | None = None,
    include_unassigned: bool = False,
    include_excluded: bool = False,
) -> list[dict]:
    """Return eligible works without altering source records or memberships.

    A coauthored work is counted once if at least one selected member is
    eligible in its publication year. In the scoped result, only eligible
    members receive individual credit. Both year bounds are inclusive.
    """
    by_id = {member["id"]: member for member in members}
    selected = set(selected_ids)
    eligible_works = []

    for original in works:
        if original["excluded"] and not include_excluded:
            continue

        year = original["year"]
        if global_from is not None and (year is None or year < global_from):
            continue
        if global_to is not None and (year is None or year > global_to):
            continue

        credited_ids = []
        for member_id in original["member_ids"]:
            if member_id not in selected or member_id not in by_id:
                continue
            member = by_id[member_id]
            start = member["publication_year_from"]
            end = member["publication_year_to"]
            if start is not None and (year is None or year < start):
                continue
            if end is not None and (year is None or year > end):
                continue
            credited_ids.append(member_id)

        if not credited_ids and not (
            include_unassigned and not original["member_ids"]
        ):
            continue

        result = dict(original)
        result["member_ids"] = credited_ids
        result["team_members"] = "; ".join(
            by_id[member_id]["name"] for member_id in credited_ids
        )
        result["team_member_count"] = len(credited_ids)
        eligible_works.append(result)

    return eligible_works


def list_members(connection, include_inactive: bool = True) -> list[dict]:
    """Return team members with their OpenAlex IDs and reporting-year bounds."""
    condition = "" if include_inactive else "WHERE m.active=1"
    members = [
        dict(row)
        for row in connection.execute(
            f"""SELECT m.* FROM members m {condition}
                ORDER BY m.last_name COLLATE NOCASE,
                         m.first_name COLLATE NOCASE"""
        )
    ]

    author_ids = defaultdict(list)
    for row in connection.execute(
        "SELECT member_id, openalex_id FROM author_ids ORDER BY openalex_id"
    ):
        author_ids[row["member_id"]].append(row["openalex_id"])

    for member in members:
        member["name"] = (
            member["first_name"] + " " + member["last_name"]
        ).strip()
        member["openalex_ids"] = author_ids[member["id"]]
    return members


def add_member(
    connection,
    first_name: str,
    last_name: str,
    author_ids: str | Iterable[str],
    publication_year_from=None,
    publication_year_to=None,
) -> int:
    """Create a team member and connect any existing OpenAlex source records."""
    first, last = first_name.strip(), last_name.strip()
    start, end = _validated_years(publication_year_from, publication_year_to)
    if not first or not last:
        raise ValueError("First and last names are required")

    identifiers = parse_author_ids(author_ids)
    with connection:
        member_id = connection.execute(
            """INSERT INTO members (
                   first_name, last_name, publication_year_from, publication_year_to
               ) VALUES (?,?,?,?)""",
            (first, last, start, end),
        ).lastrowid
        for identifier in identifiers:
            connection.execute(
                "INSERT INTO author_ids(openalex_id, member_id) VALUES (?,?)",
                (identifier, member_id),
            )
        _reindex_variant_members(connection, member_id)
        reindex_bibtex_members(connection)
    return member_id


def update_member(
    connection,
    member_id: int,
    first_name: str,
    last_name: str,
    ids: str | Iterable[str],
    active: bool = True,
    publication_year_from=None,
    publication_year_to=None,
) -> None:
    """Update a member and rebuild author associations after ID changes."""
    first, last = first_name.strip(), last_name.strip()
    start, end = _validated_years(publication_year_from, publication_year_to)
    if not first or not last:
        raise ValueError("First and last names are required")

    new_ids = set(parse_author_ids(ids))
    old_ids = {
        row[0] for row in connection.execute(
            "SELECT openalex_id FROM author_ids WHERE member_id=?", (member_id,)
        )
    }
    with connection:
        connection.execute(
            """UPDATE members SET
                   first_name=?, last_name=?, active=?,
                   publication_year_from=?, publication_year_to=?
               WHERE id=?""",
            (first, last, int(active), start, end, member_id),
        )
        for author_id in old_ids - new_ids:
            connection.execute(
                "DELETE FROM author_ids WHERE openalex_id=? AND member_id=?",
                (author_id, member_id),
            )
            connection.execute(
                "DELETE FROM variant_members WHERE member_id=? AND evidence=?",
                (member_id, f"openalex:{author_id}"),
            )
        for author_id in new_ids - old_ids:
            connection.execute(
                "INSERT INTO author_ids(openalex_id, member_id) VALUES (?,?)",
                (author_id, member_id),
            )
        _reindex_variant_members(connection, member_id)
        reindex_bibtex_members(connection)


def delete_member(connection, member_id: int) -> None:
    """Remove a member and their links; preserve all canonical publications."""
    with connection:
        connection.execute("DELETE FROM members WHERE id=?", (member_id,))
        reindex_bibtex_members(connection)


def _ids_by_member(connection) -> dict[str, int]:
    """Map OpenAlex Author IDs to their registered internal team-member IDs."""
    return {
        row["openalex_id"]: row["member_id"]
        for row in connection.execute("SELECT * FROM author_ids")
    }


def _raw_author_ids(raw: dict) -> set[str]:
    """Extract recognized OpenAlex Author IDs from a Work's authorships."""
    identifiers = set()
    for authorship in raw.get("authorships") or []:
        author = authorship.get("author") or {}
        try:
            identifiers.add(normalized_author_id(author.get("id") or ""))
        except ValueError:
            continue
    return identifiers


def _reindex_variant_members(connection, member_id: int) -> None:
    """Connect existing OpenAlex variants to a newly configured author alias."""
    aliases = {
        row[0] for row in connection.execute(
            "SELECT openalex_id FROM author_ids WHERE member_id=?", (member_id,)
        )
    }
    if not aliases:
        return

    for variant in connection.execute(
        "SELECT id, raw_json FROM variants WHERE source='openalex'"
    ):
        try:
            found = _raw_author_ids(json.loads(variant["raw_json"]))
        except (TypeError, ValueError):
            continue
        for author_id in aliases.intersection(found):
            connection.execute(
                """INSERT OR IGNORE INTO variant_members (
                       variant_id, member_id, evidence
                   ) VALUES (?,?,?)""",
                (variant["id"], member_id, f"openalex:{author_id}"),
            )


def add_records(
    connection,
    records: Iterable[dict],
    member_ids: Iterable[int] = (),
    evidence: str = "manual",
) -> dict:
    """Import records and attach recognized team coauthors.

    OpenAlex Work authorships are checked against every known team ID, not
    just the author whose profile was synchronized. Manual Crossref/BibTeX
    attribution is applied only to the selected members.
    """
    records = list(records)
    valid = {member["id"] for member in list_members(connection)}
    selected = set(member_ids)
    if not selected.issubset(valid):
        raise ValueError("Unknown team member")

    counts = upsert_records(connection, records)
    author_map = _ids_by_member(connection)
    with connection:
        for record in records:
            source = str(record.get("source") or "manual")
            source_id = str(record.get("source_id") or "")
            variant = connection.execute(
                "SELECT id FROM variants WHERE source=? AND source_id=?",
                (source, source_id),
            ).fetchone()
            if variant is None:
                continue

            if source == "openalex":
                found = _raw_author_ids(record.get("raw") or {})
                for author_id in found.intersection(author_map):
                    connection.execute(
                        """INSERT OR IGNORE INTO variant_members (
                               variant_id, member_id, evidence
                           ) VALUES (?,?,?)""",
                        (variant["id"], author_map[author_id], f"openalex:{author_id}"),
                    )

            for member_id in selected:
                connection.execute(
                    """INSERT OR IGNORE INTO variant_members (
                           variant_id, member_id, evidence
                       ) VALUES (?,?,?)""",
                    (variant["id"], member_id, evidence),
                )
        if any(record.get("source") in {"bibtex", "scholar_bibtex"} for record in records):
            reindex_bibtex_members(connection)
    return counts


def _automatic_member_map(connection) -> dict[int, set[int]]:
    """Collect memberships inferred from source records, before corrections."""
    mapping = defaultdict(set)
    rows = connection.execute(
        """SELECT DISTINCT v.work_id AS work_id, vm.member_id AS member_id
           FROM variant_members vm
           JOIN variants v ON v.id=vm.variant_id"""
    )
    for row in rows:
        mapping[row["work_id"]].add(row["member_id"])
    return mapping


def _member_map(connection) -> dict[int, set[int]]:
    """Apply human inclusion and exclusion decisions over inferred memberships."""
    mapping = _automatic_member_map(connection)
    for row in connection.execute(
        "SELECT work_id, member_id, decision FROM member_work_overrides"
    ):
        if row["decision"] == "include":
            mapping[row["work_id"]].add(row["member_id"])
        else:
            mapping[row["work_id"]].discard(row["member_id"])
    return mapping


def get_team_works(connection, include_excluded: bool = False) -> list[dict]:
    """Join canonical publications to their effective team-member assignments."""
    names = {member["id"]: member["name"] for member in list_members(connection)}
    mapping = _member_map(connection)
    works = get_works(connection, include_excluded=include_excluded)
    for work in works:
        member_ids = sorted(
            mapping.get(work["id"], set()),
            key=lambda identifier: names.get(identifier, ""),
        )
        work["member_ids"] = member_ids
        work["team_members"] = "; ".join(
            names.get(identifier, f"#{identifier}") for identifier in member_ids
        )
        work["team_member_count"] = len(member_ids)
    return works


def set_work_members(
    connection, work_id: int, member_ids: Iterable[int],
) -> None:
    """Persist a manual author list, including explicit removals of wrong matches."""
    desired = {int(identifier) for identifier in member_ids}
    valid = {member["id"] for member in list_members(connection)}
    if not desired.issubset(valid):
        raise ValueError("A selected team member does not exist")

    # Store explicit decisions for *all* currently registered members. A later
    # alias refresh may discover more automatic evidence, but it must never
    # silently override a human-reviewed contributor list.
    with connection:
        connection.execute(
            "DELETE FROM member_work_overrides WHERE work_id=?", (work_id,)
        )
        connection.executemany(
            """INSERT INTO member_work_overrides
               (work_id, member_id, decision) VALUES (?,?,?)""",
            [(work_id, member_id, "include" if member_id in desired else "exclude")
             for member_id in valid],
        )


def merge_team_works(connection, destination: int, others: list[int]) -> None:
    """Merge canonical works and carry forward their manual member corrections.

    If two merged works have conflicting explicit decisions for one person,
    exclusion wins to avoid falsely crediting that researcher.
    """
    from .core import merge_works

    if destination in others:
        raise ValueError("Destination publication cannot be among the sources")
    with connection:
        for source in set(others):
            overrides = connection.execute(
                """SELECT member_id, decision FROM member_work_overrides
                   WHERE work_id=?""",
                (source,),
            ).fetchall()
            for override in overrides:
                previous = connection.execute(
                    """SELECT decision FROM member_work_overrides
                       WHERE work_id=? AND member_id=?""",
                    (destination, override["member_id"]),
                ).fetchone()
                exclude = (
                    override["decision"] == "exclude"
                    or (previous is not None and previous["decision"] == "exclude")
                )
                decision = "exclude" if exclude else "include"
                connection.execute(
                    """INSERT INTO member_work_overrides (
                           work_id, member_id, decision
                       ) VALUES (?,?,?)
                       ON CONFLICT(work_id, member_id)
                       DO UPDATE SET decision=excluded.decision""",
                    (destination, override["member_id"], decision),
                )
        merge_works(connection, destination, others)



def summarize_member_impact(
    works: list[dict],
    selected_member_ids: Iterable[int],
    labels: dict[int, str],
) -> list[dict]:
    """Summarize credited publications and indexed citations for each member.

    ``works`` must already have been filtered by ``scope_team_works``. In
    particular, its ``member_ids`` contain only selected contributors whose
    individual publication-year windows include that work. The citation count
    of a shared publication is credited in full to each eligible coauthor,
    but only once to any one researcher. Consequently, totals summed across
    researchers can exceed the deduplicated team's publication/citation totals.

    Members with no eligible publications are included with zero values.
    """
    summaries = {
        member_id: {
            "Researcher": labels[member_id],
            "Publications": 0,
            "Citations": 0,
        }
        for member_id in dict.fromkeys(selected_member_ids)
        if member_id in labels
    }

    for work in works:
        citations = int(work.get("citations") or 0)
        for member_id in set(work.get("member_ids") or ()):
            if member_id not in summaries:
                continue
            summaries[member_id]["Publications"] += 1
            summaries[member_id]["Citations"] += citations

    return list(summaries.values())


def team_csv_export(works: list[dict]) -> bytes:
    """Export one row per canonical work, including credited team members."""
    columns = [
        "id", "title", "authors", "year", "venue", "bibtex_type", "work_type",
        "doi", "volume", "number", "pages", "month", "publication_date",
        "publisher", "issn", "isbn", "url", "edition",
        "citations", "citation_source", "openalex_id", "team_members",
        "team_member_count", "sources", "variant_count",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(works)
    return buffer.getvalue().encode("utf-8-sig")


def team_json_export(connection) -> bytes:
    """Create a full provenance archive for inspection (not an import format)."""
    works = get_team_works(connection, include_excluded=True)
    for work in works:
        work["variants"] = get_variants(connection, work["id"])

    payload = {
        "format": "paperazzi-v1",
        "members": list_members(connection),
        "works": works,
        "member_work_overrides": [
            dict(row) for row in connection.execute(
                "SELECT * FROM member_work_overrides"
            )
        ],
        "variant_members": [
            dict(row) for row in connection.execute("SELECT * FROM variant_members")
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
