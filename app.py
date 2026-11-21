"""Streamlit interface for Paperazzi's local team bibliography.

The interface is intentionally thin: bibliographic normalization lives in
``paperazzi.core``, source adapters in ``paperazzi.sources``, and team-level
attribution and reporting rules in ``paperazzi.team``.

All edits are committed to the local SQLite database before Streamlit reruns.
No background synchronization or network calls take place automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from functools import partial
from pathlib import Path

import pandas as pd
from pandas.io.formats.style import Styler
import plotly.express as px
import streamlit as st

from paperazzi import sources
from paperazzi.ui_selection import selected_row_identifier
from paperazzi.core import (
    BIBTEX_TYPES,
    bibtex_export,
    fuzzy_pairs,
    get_variants,
    h_index,
    normalize_openalex,
    set_bibtex_type_override,
    set_excluded,
    set_override,
    split_variant,
)
from paperazzi.team import (
    add_member,
    add_records,
    connect_team,
    delete_member,
    get_team_works,
    list_members,
    member_import_bounds,
    merge_team_works,
    scope_team_works,
    summarize_member_impact,
    set_work_members,
    team_csv_export,
    team_json_export,
    update_member,
)

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("PAPERAZZI_DB", str(ROOT / "data" / "paperazzi.sqlite")))

# A category is a reporting view, not a database field. The canonical field is
# always the BibTeX entry type, which is retained across imports and exports.
CATEGORIES = {
    "All": lambda work: True,
    "Articles": lambda work: work["bibtex_type"] == "article",
    "Proceedings": lambda work: work["bibtex_type"] in {"inproceedings", "conference"},
    "Book Chapters": lambda work: work["bibtex_type"] == "incollection",
    "Patents": lambda work: work["bibtex_type"] == "patent",
    "Other": lambda work: work["bibtex_type"]
    not in {"article", "inproceedings", "conference", "incollection", "patent"},
}

STYLE = """
<style>
    .block-container { max-width: 1500px; padding-top: 1.4rem; }
    h1, h2, h3 { letter-spacing: -.022em; }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, .22);
        border-radius: 12px;
        padding: 10px 15px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, .16);
    }
</style>
"""


def report_and_reload(message: str, *, clear_duplicate_suggestions: bool = True) -> None:
    """Refresh database-backed views without disrupting the current tab.

    A source import invalidates earlier duplicate searches, while a merge or
    split recomputes the suggestions before calling this function. In that
    case, retaining the refreshed list allows successive merges without
    asking the user to launch the duplicate search again.
    """
    if clear_duplicate_suggestions:
        st.session_state.pop("duplicate_suggestions", None)
    st.session_state["report"] = message
    st.rerun()


def refresh_duplicate_suggestions(connection, members: list[dict], options: dict) -> None:
    """Recompute an existing duplicate search within the current report scope.

    The same global years, individual member ranges and membership filters
    apply before and after a merge, so old publications do not unexpectedly
    appear among the next candidate pairs.
    """
    if "duplicate_suggestions" not in st.session_state:
        return

    current_works = get_team_works(connection, include_excluded=True)
    scoped_works = scope_team_works(
        current_works,
        members,
        options["selected_members"],
        options["year_from"],
        options["year_to"],
        include_unassigned=options["include_unassigned"],
    )
    threshold = st.session_state.get("reconcile_threshold", 0.82)
    st.session_state["duplicate_suggestions"] = fuzzy_pairs(
        scoped_works[:1200], threshold, limit=100
    )


def short_label(work: dict) -> str:
    """Return a compact, unambiguous label for publication selectors."""
    return f"#{work['id']} · {work['year'] or '?'} · {work['title'][:100]}"


def remember_table_selection(
    table_key: str,
    record_ids: tuple[int, ...],
    destination_key: str,
) -> None:
    """Remember a selected database ID when a dataframe selection changes.

    A clicked cell supplies its original dataframe row position, even if the
    user sorted the table visually. We map that position to a persistent ID
    and keep it in session state. The callback runs before the next rendering
    pass, so the correct editor opens immediately.
    """
    identifier = selected_row_identifier(st.session_state.get(table_key), record_ids)
    if identifier is not None:
        st.session_state[destination_key] = identifier


def category_works(works: list[dict], category: str) -> list[dict]:
    """Filter canonical works using the same categories across all views."""
    return [work for work in works if CATEGORIES[category](work)]


def highlight_selected_row(
    table: pd.DataFrame,
    record_id: int | None,
    id_column: str,
) -> "Styler":
    """Emphasize the complete row whose persistent database ID is selected.

    Streamlit's single-cell selection makes every cell clickable without a
    checkbox column. Applying a Pandas Styler gives the *whole row* a visible
    background, not just the clicked cell. The identifier (rather than a row
    position) remains stable when the table is sorted or filtered.
    """
    highlight = "background-color: #f7dce2; color: #461b26; font-weight: 600"

    def style_row(row: pd.Series) -> list[str]:
        selected = record_id is not None and row[id_column] == record_id
        return [highlight if selected else "" for _ in row]

    return table.style.apply(style_row, axis=1)


def members_table(members: list[dict]) -> pd.DataFrame:
    """Build a readable table of team members and their optional year bounds."""
    columns = [
        "Internal ID", "First name", "Last name", "OpenAlex IDs",
        "Start year", "End year", "Active",
    ]
    records = [
        {
            "Internal ID": person["id"],
            "First name": person["first_name"],
            "Last name": person["last_name"],
            "OpenAlex IDs": ", ".join(person["openalex_ids"]),
            "Start year": person["publication_year_from"],
            "End year": person["publication_year_to"],
            "Active": "Yes" if person["active"] else "No",
        }
        for person in members
    ]
    return pd.DataFrame(records, columns=columns)


def render_sidebar(members: list[dict], member_labels: dict[int, str]) -> dict:
    """Collect API credentials and non-destructive report-scoping options."""
    with st.sidebar:
        st.title("📸 Paperazzi")
        st.caption("Team publication tracking · local data")

        api_key = st.text_input(
            "OpenAlex API key",
            type="password",
            value=os.getenv("OPENALEX_API_KEY", ""),
            help="Used only for API requests; never stored in SQLite.",
        )
        email = st.text_input(
            "API contact email (optional)",
            value=os.getenv("PAPERAZZI_EMAIL", ""),
            help="Optional contact address for OpenAlex and Crossref API requests.",
        )

        st.divider()
        st.markdown("**Reporting scope**")
        all_years = st.checkbox("All years", value=False)
        year_from, year_to = (None, None)
        if not all_years:
            year_from, year_to = st.slider(
                "Publication years", 1990, 2035, (2022, 2026)
            )

        selected_members = st.multiselect(
            "Team members",
            options=list(member_labels),
            default=[person["id"] for person in members],
            format_func=lambda identifier: member_labels[identifier],
            help=(
                "Former members are included if their publication year is eligible. "
                "A coauthored publication appears only once."
            ),
        )
        include_unassigned = st.checkbox("Include unassigned publications", value=False)

        st.divider()
        st.caption("Local SQLite database")
        st.code(str(DB_PATH), language=None)
        st.caption("Never commit an API key or your local database to GitHub.")

    return {
        "api_key": api_key,
        "email": email,
        "all_years": all_years,
        "year_from": year_from,
        "year_to": year_to,
        "selected_members": selected_members,
        "include_unassigned": include_unassigned,
    }


def render_team_tab(connection, members: list[dict], labels: dict[int, str]) -> None:
    """Manage individual OpenAlex author identifiers and inclusive year bounds."""
    st.subheader("Team members")
    st.write(
        "Each researcher may have **multiple OpenAlex Author IDs**. "
        "Coauthored publications are linked to all recognized team members."
    )
    st.caption(
        "Start and end years are optional and inclusive. A blank field means "
        "no limit. The active/inactive flag does not remove former members "
        "from historical reporting."
    )
    st.caption("Click anywhere on a row to highlight it and edit that team member below.")
    member_ids = tuple(person["id"] for person in members)
    st.dataframe(
        highlight_selected_row(
            members_table(members),
            st.session_state.get("selected_member_id"),
            "Internal ID",
        ),
        hide_index=True,
        use_container_width=True,
        key="team_member_table",
        on_select=partial(
            remember_table_selection,
            "team_member_table",
            member_ids,
            "selected_member_id",
        ),
        selection_mode="single-cell",
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Add a member")
        with st.form("add_member_form"):
            first_name = st.text_input("First name")
            last_name = st.text_input("Last name")
            author_ids = st.text_input(
                "OpenAlex Author ID(s)", placeholder="A5028479581, A1234567890"
            )
            st.caption("Inclusive publication years; leave blank for no limit.")
            start_year = st.number_input(
                "Start year (optional)",
                min_value=1000,
                max_value=2200,
                value=None,
                step=1,
                placeholder="No limit",
                key="add_start",
            )
            end_year = st.number_input(
                "End year (optional)",
                min_value=1000,
                max_value=2200,
                value=None,
                step=1,
                placeholder="No limit",
                key="add_end",
            )
            submitted = st.form_submit_button("Add researcher", type="primary")

        if submitted:
            try:
                add_member(connection, first_name, last_name, author_ids, start_year, end_year)
                report_and_reload("Researcher added.")
            except Exception as error:
                st.error(str(error))

    with right:
        st.markdown("#### Edit a member")
        if not members:
            st.info("Add a researcher to begin.")
            return

        edit_id = st.session_state.get("selected_member_id")
        person = next((item for item in members if item["id"] == edit_id), None)
        if person is None:
            st.info("Select a team member in the table above to edit their details.")
            return

        st.caption(f"Editing: {person['name']}")
        with st.form(f"edit_member_{edit_id}"):
            first_name = st.text_input("First name", value=person["first_name"])
            last_name = st.text_input("Last name", value=person["last_name"])
            author_ids = st.text_area(
                "OpenAlex Author IDs (comma-separated)",
                value=", ".join(person["openalex_ids"]),
                height=90,
            )
            st.caption("Inclusive publication years; leave blank for no limit.")
            start_year = st.number_input(
                "Start year (optional)",
                min_value=1000,
                max_value=2200,
                value=person["publication_year_from"],
                step=1,
                placeholder="No limit",
                key=f"edit_start_{edit_id}",
            )
            end_year = st.number_input(
                "End year (optional)",
                min_value=1000,
                max_value=2200,
                value=person["publication_year_to"],
                step=1,
                placeholder="No limit",
                key=f"edit_end_{edit_id}",
            )
            active = st.checkbox(
                "Currently active member",
                value=bool(person["active"]),
                help="Status is informational; historical inclusion depends on year bounds.",
            )
            saved = st.form_submit_button("Save changes")

        if saved:
            try:
                update_member(
                    connection,
                    edit_id,
                    first_name,
                    last_name,
                    author_ids,
                    active,
                    start_year,
                    end_year,
                )
                report_and_reload("Researcher updated.")
            except Exception as error:
                st.error(str(error))

        with st.expander("Remove this member"):
            st.warning(
                "Publication records will remain in the library, but their "
                "associations with this member will be removed."
            )
            confirmed = st.checkbox(
                "I confirm that this member should be removed",
                key=f"confirm_delete_{edit_id}",
            )
            if st.button("Remove member", disabled=not confirmed, key="delete_member"):
                delete_member(connection, edit_id)
                st.session_state.pop("selected_member_id", None)
                report_and_reload("Researcher removed; publications retained.")


def synchronize_openalex(
    connection,
    members: list[dict],
    member_ids: list[int],
    api_key: str,
    email: str,
    max_works: int,
    global_from: int | None,
    global_to: int | None,
) -> None:
    """Import each selected author independently and preserve successful imports."""
    tasks = [
        (person, author_id)
        for person in members
        if person["id"] in member_ids
        for author_id in person["openalex_ids"]
    ]
    progress = st.progress(0.0)
    status = st.empty()
    failures: list[str] = []
    skipped: list[str] = []
    totals = Counter()

    for position, (person, author_id) in enumerate(tasks, start=1):
        status.info(
            f"OpenAlex: {person['name']} — {author_id} "
            f"({position}/{len(tasks)})"
        )
        try:
            lower, upper, empty = member_import_bounds(person, global_from, global_to)
            if empty:
                skipped.append(f"{person['name']} ({author_id}): no overlapping years")
                continue

            records = sources.openalex_works(
                author_id=author_id,
                api_key=api_key,
                email=email,
                max_works=max_works,
                year_from=lower,
                year_to=upper,
            )
            # OpenAlex authorships can match other members besides the query author.
            totals.update(add_records(connection, records))
            totals["retrieved"] += len(records)
        except Exception as error:
            failures.append(f"{person['name']} ({author_id}): {error}")
        finally:
            progress.progress(position / len(tasks))

    status.empty()
    warnings = []
    if skipped:
        warnings.append("Skipped profiles (no overlapping years):\n\n" + "\n\n".join(skipped))
    if failures:
        warnings.append(
            "Individual imports failed; successful imports were kept:\n\n"
            + "\n\n".join(failures)
        )
    st.session_state["sync_warnings"] = warnings
    report_and_reload(
        f"{totals['retrieved']} records retrieved; "
        f"{totals['new_works']} publications created, "
        f"{totals['new_variants']} source records added, "
        f"and {totals['updated_variants']} updated."
    )


def render_import_tab(connection, members: list[dict], labels: dict, options: dict) -> None:
    """Provide OpenAlex, BibTeX, Crossref, and title/DOI import workflows."""
    st.subheader("Import publications")
    st.caption(
        "Records are automatically matched by DOI, OpenAlex Work ID, or an "
        "exactly normalized title and publication year. Ambiguous matches require review."
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### OpenAlex · team synchronization")
        eligible = [person["id"] for person in members if person["openalex_ids"]]
        import_members = st.multiselect(
            "Researchers to synchronize",
            eligible,
            default=eligible,
            format_func=lambda identifier: labels[identifier],
            key="import_members",
        )
        limit = st.number_input(
            "Maximum records per Author ID", 100, 10000, 2000, step=100
        )
        limit_period = st.checkbox(
            "Also limit imports to the team-wide reporting period",
            value=False,
            disabled=options["all_years"],
            help=(
                "Individual member year bounds always apply; the global period "
                "is an optional additional restriction during import."
            ),
        )
        st.caption(
            "Each member's configured publication-year bounds are always "
            "applied. The global reporting period is optional during import."
        )

        if st.button("Synchronize OpenAlex", type="primary", use_container_width=True):
            if not options["api_key"].strip():
                st.error("Enter your OpenAlex API key in the sidebar.")
            elif not import_members:
                st.warning("Select at least one researcher.")
            else:
                synchronize_openalex(
                    connection,
                    members,
                    import_members,
                    options["api_key"],
                    options["email"],
                    int(limit),
                    options["year_from"] if limit_period else None,
                    options["year_to"] if limit_period else None,
                )

    with right:
        st.markdown("#### BibTeX · manual import")
        uploaded_files = st.file_uploader(
            "BibTeX files (.bib or .txt)",
            type=["bib", "txt"],
            accept_multiple_files=True,
        )
        bibtex_members = st.multiselect(
            "Assign every record to these team members (optional)",
            list(labels),
            format_func=lambda identifier: labels[identifier],
            key="bib_members",
            help=(
                "Selecting multiple members assigns every entry in each file "
                "to all of them. Leave blank for mixed bibliographies and review "
                "author assignments after import."
            ),
        )

        if st.button("Import files", disabled=not uploaded_files, use_container_width=True):
            summaries = []
            for uploaded in uploaded_files:
                try:
                    content = uploaded.getvalue()
                    # A stable per-file namespace prevents BibTeX key collisions.
                    fingerprint = hashlib.sha1(
                        uploaded.name.encode("utf-8") + b"\0" + content
                    ).hexdigest()[:12]
                    records = sources.parse_bibtex(content, namespace=fingerprint)
                    counts = add_records(
                        connection, records, member_ids=bibtex_members, evidence="bibtex"
                    )
                    summaries.append(
                        f"{uploaded.name}: {len(records)} records, "
                        f"{counts['new_works']} new publications"
                    )
                except Exception as error:
                    st.error(f"{uploaded.name}: {error}")
            if summaries:
                report_and_reload(". ".join(summaries))

        st.info(
            "BibTeX entries such as **@incollection** (book chapters) and "
            "**@patent** (patents) are supported."
        )

    st.divider()
    st.markdown("#### Crossref · supplementary search")
    column_a, column_b = st.columns([1.3, 2])

    with column_a:
        member_id = (
            st.selectbox(
                "Researcher",
                list(labels),
                format_func=lambda identifier: labels[identifier],
                key="crossref_member",
            )
            if members else None
        )
        max_results = st.number_input("Maximum results", 20, 1000, 200, step=20)
        if st.button("Search Crossref by author", disabled=member_id is None):
            try:
                person = next(person for person in members if person["id"] == member_id)
                st.session_state["crossref_results"] = sources.crossref_author_search(
                    person["name"],
                    options["email"],
                    rows=int(max_results),
                    family_name=person["last_name"],
                )
                st.session_state["crossref_for"] = member_id
            except Exception as error:
                st.error(f"Crossref: {error}")

    with column_b:
        results = st.session_state.get("crossref_results", [])
        st.caption(
            "Crossref searches by name. Review each match before importing "
            "to avoid publications by namesakes."
        )
        if results:
            choices = {
                index: f"{record.get('year') or '?'} · {record['title'][:110]}"
                for index, record in enumerate(results)
            }
            st.dataframe(
                pd.DataFrame([
                    {"Year": record["year"], "Title": record["title"], "DOI": record["doi"]}
                    for record in results
                ]),
                hide_index=True,
                height=220,
                use_container_width=True,
            )
            selected = st.multiselect(
                "Records to import",
                list(choices),
                format_func=lambda index: choices[index],
                key="crossref_selected",
            )
            if st.button("Import selection", disabled=not selected):
                member_id = st.session_state.get("crossref_for")
                counts = add_records(
                    connection,
                    [results[index] for index in selected],
                    member_ids=[member_id],
                    evidence="crossref",
                )
                report_and_reload(
                    f"Crossref: {len(selected)} records imported; "
                    f"{counts['new_works']} new publications."
                )
        else:
            st.caption("No search results loaded.")

    st.divider()
    st.markdown("#### Find a publication by title or DOI")
    query = st.text_input("Search OpenAlex", placeholder="Title or DOI…")
    if st.button("Search OpenAlex", disabled=not query.strip()):
        try:
            st.session_state["title_results"] = sources.openalex_title_search(
                query, options["api_key"], max_results=20
            )
        except Exception as error:
            st.error(f"OpenAlex search: {error}")

    found = st.session_state.get("title_results", [])
    if found:
        choices = {
            index: f"{record.get('year') or '?'} · {record['title'][:120]}"
            for index, record in enumerate(found)
        }
        selected = st.multiselect(
            "Missing publications", list(choices),
            format_func=lambda index: choices[index], key="title_selection",
        )
        if st.button("Import selected records", disabled=not selected):
            counts = add_records(connection, [found[index] for index in selected])
            report_and_reload(
                f"{len(selected)} records added; {counts['new_works']} new publications."
            )


def render_library_tab(
    connection, works: list[dict], members: list[dict], labels: dict, options: dict
) -> None:
    """Display category lists and editable canonical publication metadata."""
    st.subheader("Consolidated library")
    st.caption(
        "Filtered by report years, individual year bounds, and selected members. "
        "Coauthors outside the period remain in the database but are not counted."
    )
    search = st.text_input(
        "Search title, author, DOI, or team member",
        placeholder="retina, representation learning…",
    )
    show_excluded = st.checkbox("Include excluded publications", value=False)

    scoped = scope_team_works(
        works,
        members,
        options["selected_members"],
        options["year_from"],
        options["year_to"],
        include_unassigned=options["include_unassigned"],
        include_excluded=show_excluded,
    )
    searchable_fields = ("title", "authors", "doi", "team_members", "venue")
    filtered = [
        work for work in scoped
        if not search or search.casefold() in " ".join(
            str(work.get(field) or "") for field in searchable_fields
        ).casefold()
    ]

    st.caption("Click any cell to highlight its entire publication row and open the editor below.")
    tab_names = [
        f"{category} ({len(category_works(filtered, category))})"
        for category in CATEGORIES
    ]
    # Persist the active category when a row click triggers a Streamlit rerun.
    category_tabs = st.tabs(
        tab_names, key="paperazzi_library_category", on_change="rerun"
    )
    for tab, category in zip(category_tabs, CATEGORIES):
        if not tab.open:
            continue
        with tab:
            subset = category_works(filtered, category)
            record_ids = tuple(work["id"] for work in subset)
            table_key = f"library_table_{category.lower().replace(' ', '_')}"
            table = pd.DataFrame([
                {
                    "ID": work["id"],
                    "Year": work["year"],
                    "BibTeX type": work["bibtex_type"],
                    "Title": work["title"],
                    "Team members": work["team_members"],
                    "Citations*": work["citations"],
                    "DOI": work["doi"],
                    "Sources": work["sources"],
                    "Excluded": "Yes" if work["excluded"] else "No",
                }
                for work in subset
            ])
            st.dataframe(
                highlight_selected_row(
                    table, st.session_state.get("selected_work_id"), "ID"
                ),
                use_container_width=True,
                hide_index=True,
                height=330,
                key=table_key,
                on_select=partial(
                    remember_table_selection,
                    table_key,
                    record_ids,
                    "selected_work_id",
                ),
                selection_mode="single-cell",
            )
            if subset:
                st.download_button(
                    f"↓ Export {category} CSV",
                    team_csv_export(subset),
                    file_name=f"paperazzi_{category.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"csv_{category}",
                )

    st.divider()
    st.markdown("#### Publication details and corrections")
    if not filtered:
        st.info("No publications match the current filters. Adjust your scope or import data.")
        return

    publication_id = st.session_state.get("selected_work_id")
    work = next((item for item in filtered if item["id"] == publication_id), None)
    if work is None:
        st.info(
            "Select a publication in one of the tables above to edit it. "
            "If a previously selected publication is hidden, adjust the filters."
        )
        return

    original = next(item for item in works if item["id"] == publication_id)
    st.caption(f"Editing publication #{publication_id}")

    st.markdown(f"**{work['title']}**")
    st.caption(
        f"{work['year'] or '?'} · @{work['bibtex_type']} · "
        f"{work.get('venue') or 'Unknown venue'}"
    )
    st.write(work.get("authors") or "Authors not available")

    with st.form(f"edit_work_{publication_id}"):
        title_col, year_col, type_col = st.columns([3, 1, 1.5])
        revised_title = title_col.text_input(
            "Corrected title (blank = automatic)", value=work.get("title_override") or ""
        )
        revised_year = year_col.number_input(
            "Corrected year (0 = automatic)",
            0, 2100,
            value=work.get("year_override") or 0,
        )
        available_types = ["Automatic"] + sorted(set(BIBTEX_TYPES) - {"conference"})
        current_type = work.get("bibtex_type_override") or "Automatic"
        revised_type = type_col.selectbox(
            "BibTeX type",
            available_types,
            index=available_types.index(current_type) if current_type in available_types else 0,
        )
        credited_members = st.multiselect(
            "Contributing team members (all years)",
            list(labels),
            default=original["member_ids"],
            format_func=lambda identifier: labels[identifier],
            help=(
                "This edits the underlying author assignments independently "
                "of the current reporting period. Corrections survive reimports."
            ),
        )
        saved = st.form_submit_button("Save corrections", type="primary")

    if saved:
        set_override(connection, publication_id, revised_title, revised_year or None)
        set_bibtex_type_override(
            connection,
            publication_id,
            None if revised_type == "Automatic" else revised_type,
        )
        set_work_members(connection, publication_id, credited_members)
        report_and_reload("Publication updated.")

    left, right = st.columns(2)
    if work.get("doi"):
        left.link_button(
            "Open DOI", f"https://doi.org/{work['doi']}", use_container_width=True
        )
    if work.get("openalex_id"):
        right.link_button(
            "Open in OpenAlex",
            "https://openalex.org/works/" + normalize_openalex(work["openalex_id"]),
            use_container_width=True,
        )

    action = "Restore" if work["excluded"] else "Exclude from report"
    if st.button(action, key="exclude_work"):
        set_excluded(connection, publication_id, not work["excluded"])
        st.rerun()

    st.markdown("**Original source records (retained after merging)**")
    records = get_variants(connection, publication_id)
    st.dataframe(
        pd.DataFrame([
            {
                "Record": record["id"],
                "Source": record["source"],
                "Title": record["title"],
                "Year": record["year"],
                "Type": record["bibtex_type"],
                "DOI": record["doi"],
                "Source ID": record["source_id"],
            }
            for record in records
        ]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "* Citation counts are selected from the highest-priority available "
        "source and are never summed across sources."
    )


def render_reconcile_tab(
    connection, works: list[dict], members: list[dict], options: dict
) -> None:
    """Suggest fuzzy matches while requiring manual confirmation of merges."""
    st.subheader("Reconcile source records")
    st.write(
        "Identical DOIs and OpenAlex Work IDs are matched automatically. "
        "Similar titles are suggestions only and require human review."
    )

    threshold = st.slider(
        "Title similarity threshold", 0.65, 0.98, 0.82,
        step=0.01, key="reconcile_threshold",
    )
    candidates = works[:1200]
    if len(works) > 1200:
        st.info(
            "Suggestions examine the 1,200 most recent publications for "
            "responsiveness. Manual merging is available for the full reporting scope."
        )
    if st.button("Find potential duplicates"):
        with st.spinner("Comparing publication titles…"):
            st.session_state["duplicate_suggestions"] = fuzzy_pairs(
                candidates, threshold, limit=100
            )

    suggestions = st.session_state.get("duplicate_suggestions", [])
    st.caption(
        f"{len(suggestions)} candidate pair(s). Search again after "
        "changing the threshold or importing more publications."
    )
    if suggestions:
        candidate_labels = {
            index: f"{score:.0%} · {first['title'][:60]} ⇄ {second['title'][:60]}"
            for index, (score, first, second) in enumerate(suggestions)
        }
        index = st.selectbox(
            "Candidate pair", list(candidate_labels),
            format_func=lambda value: candidate_labels[value],
        )
        _, first, second = suggestions[index]
        left, right = st.columns(2)
        for column, work in ((left, first), (right, second)):
            column.write(f"**#{work['id']} · {work['title']}**")
            column.caption(
                f"{work['year']} · {work['doi'] or 'No DOI'} · {work['team_members']}"
            )
        destination = st.radio(
            "Canonical publication",
            [first["id"], second["id"]],
            format_func=lambda identifier: f"#{identifier}",
            horizontal=True,
        )
        if st.button("Merge candidate pair", type="primary"):
            other = second["id"] if destination == first["id"] else first["id"]
            merge_team_works(connection, destination, [other])
            refresh_duplicate_suggestions(connection, members, options)
            report_and_reload(
                "Two publications merged; member assignments retained.",
                clear_duplicate_suggestions=False,
            )

    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Manual merge")
        work_labels = {work["id"]: short_label(work) for work in works}
        selected = st.multiselect(
            "Select at least two publications",
            list(work_labels),
            format_func=lambda identifier: work_labels[identifier],
        )
        if len(selected) >= 2:
            destination = st.selectbox(
                "Canonical publication",
                selected,
                format_func=lambda identifier: work_labels[identifier],
            )
            if st.button("Confirm merge"):
                merge_team_works(
                    connection, destination,
                    [identifier for identifier in selected if identifier != destination],
                )
                refresh_duplicate_suggestions(connection, members, options)
                report_and_reload(
                    "Merge completed.", clear_duplicate_suggestions=False,
                )

    with right:
        st.markdown("#### Split a source record")
        multi_source = [work for work in works if work["variant_count"] > 1]
        if not multi_source:
            st.caption("No publications have multiple source records to split.")
            return

        split_labels = {work["id"]: short_label(work) for work in multi_source}
        parent_id = st.selectbox(
            "Publication with multiple source records",
            list(split_labels),
            format_func=lambda identifier: split_labels[identifier],
        )
        variants = get_variants(connection, parent_id)
        variant_labels = {
            variant["id"]: f"{variant['source']} · {variant['title'][:95]}"
            for variant in variants
        }
        variant_id = st.selectbox(
            "Source record to separate",
            list(variant_labels),
            format_func=lambda identifier: variant_labels[identifier],
        )
        if st.button("Split into a separate publication"):
            split_variant(connection, variant_id)
            refresh_duplicate_suggestions(connection, members, options)
            report_and_reload(
                "Source record split. Review its team-member assignments.",
                clear_duplicate_suggestions=False,
            )


def render_statistics_tab(
    connection,
    selected_works: list[dict],
    selected_members: list[int],
    labels: dict,
    year_from: int,
) -> None:
    """Summarize the currently scoped, deduplicated team bibliography."""
    st.subheader("Team statistics")
    if not selected_works:
        st.info("Import publications and check your reporting scope.")
        return

    st.caption(
        "Coauthored publications count once for the team, but contribute "
        "to each eligible coauthor's individual total."
    )
    frame = pd.DataFrame(selected_works)
    first, second, third = st.columns(3)
    first.metric("Indexed citations*", int(frame["citations"].sum()))
    second.metric("h-index*", h_index(frame["citations"]))
    third.metric("i10-index*", int((frame["citations"] >= 10).sum()))

    # First row: the breakdown by BibTeX entry type and publication year.
    left, right = st.columns(2)
    by_type = frame.groupby("bibtex_type").size().reset_index(name="Publications")
    left.plotly_chart(
        px.bar(
            by_type,
            x="bibtex_type",
            y="Publications",
            title="BibTeX entry types",
            labels={"bibtex_type": "BibTeX type"},
        ),
        use_container_width=True,
    )

    year_frame = frame.dropna(subset=["year"])
    if not year_frame.empty:
        by_year = year_frame.groupby("year").size().reset_index(name="Publications")
        right.plotly_chart(
            px.bar(
                by_year,
                x="year",
                y="Publications",
                title="Unique publications by year",
                labels={"year": "Year"},
            ),
            use_container_width=True,
        )
    else:
        right.info("Publication years are unavailable for the selected works.")

    # Second row: each eligible coauthor receives full credit for a shared
    # paper and its citations. These per-member sums are not team totals.
    member_impact = summarize_member_impact(selected_works, selected_members, labels)
    left, right = st.columns(2)
    if member_impact:
        contributions = pd.DataFrame(member_impact)
        by_publications = contributions.sort_values(
            ["Publications", "Researcher"], ascending=[False, True]
        )
        left.plotly_chart(
            px.bar(
                by_publications,
                x="Researcher",
                y="Publications",
                title="Publications by team member",
            ),
            use_container_width=True,
        )

        by_citations = contributions.sort_values(
            ["Citations", "Researcher"], ascending=[False, True]
        )
        right.plotly_chart(
            px.bar(
                by_citations,
                x="Researcher",
                y="Citations",
                title="Citations by team member",
            ),
            use_container_width=True,
        )
    else:
        left.info("No team members are selected.")
        right.info("No team members are selected.")

    # A separate category summary includes book chapters independently of 'Other'.
    st.markdown("#### Publication categories")
    columns = st.columns(4)
    for column, category in zip(columns, ("Articles", "Proceedings", "Book Chapters", "Patents")):
        column.metric(category, len(category_works(selected_works, category)))

    # OpenAlex exposes counts_by_year: the year refers to the citation event,
    # not the year in which the cited publication was originally published.
    annual_citations: Counter[int] = Counter()
    for work in selected_works:
        openalex_variant = next(
            (variant for variant in get_variants(connection, work["id"])
             if variant["source"] == "openalex"),
            None,
        )
        if openalex_variant is None:
            continue
        try:
            raw = json.loads(openalex_variant["raw_json"])
        except (TypeError, ValueError):
            continue
        for record in raw.get("counts_by_year") or []:
            year = record.get("year")
            count = record.get("cited_by_count")

            if year is None or count is None:
                continue

            year = int(year)

            # Restrict citation events to the global reporting period.
            if year_from is not None and year < year_from:
                continue

            annual_citations[year] += int(count)

    if annual_citations:
        citation_frame = pd.DataFrame([
            {"Year": year, "Citations": citations}
            for year, citations in sorted(annual_citations.items())
        ])
        fig = px.line(
            citation_frame,
            x="Year",
            y="Citations",
            markers=True,
            title="Citations received per year (OpenAlex)",
        )

        fig.update_xaxes(
            tickmode="linear",
            tick0=int(citation_frame["Year"].min()),
            dtick=1,
            tickformat="d",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Most-cited publications")
    st.dataframe(
        frame.nlargest(25, "citations")[[
            "title", "year", "citations", "bibtex_type", "team_members",
        ]].rename(columns={
            "title": "Title", "year": "Year", "citations": "Citations",
            "bibtex_type": "Type", "team_members": "Team members",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "* Citation counts, h-index, and i10-index depend on the coverage "
        "and update frequency of the external metadata sources."
    )


def render_export_tab(connection, works: list[dict], members: list[dict]) -> None:
    """Export the scoped bibliography and full database provenance separately."""
    st.subheader("Export the report")
    st.write(
        "Bibliography exports follow the global period and individual member "
        "year bounds, and exclude publications marked as excluded. "
        "The underlying author assignments are not modified."
    )

    first, second, third = st.columns(3)
    first.download_button(
        "↓ Publications · CSV",
        team_csv_export(works),
        file_name="paperazzi_publications.csv",
        mime="text/csv",
        use_container_width=True,
    )
    second.download_button(
        "↓ Publications · BibTeX",
        bibtex_export(works),
        file_name="paperazzi_publications.bib",
        mime="application/x-bibtex",
        use_container_width=True,
    )
    third.download_button(
        "↓ Full archive · JSON",
        team_json_export(connection),
        file_name="paperazzi_backup.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("#### Category-specific CSV exports")
    columns = st.columns(4)
    for column, category in zip(columns, ("Articles", "Proceedings", "Book Chapters", "Patents")):
        column.download_button(
            f"↓ {category} CSV",
            team_csv_export(category_works(works, category)),
            file_name=f"paperazzi_{category.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("#### Team roster")
    st.download_button(
        "↓ Team members CSV",
        members_table(members).to_csv(index=False).encode("utf-8-sig"),
        file_name="paperazzi_team.csv",
        mime="text/csv",
    )
    st.warning(
        "The JSON export is a reference archive, not a restore function. "
        "For a restorable backup, stop Paperazzi and copy "
        "data/paperazzi.sqlite as well."
    )


def main() -> None:
    """Open the local database and render every reporting and curation tab."""
    st.set_page_config(
        page_title="Paperazzi · Team bibliography", page_icon="📸", layout="wide"
    )
    st.markdown(STYLE, unsafe_allow_html=True)

    # Fresh installs start with an empty team; existing SQLite files are kept.
    connection = connect_team(DB_PATH)
    try:
        members = list_members(connection)
        labels = {person["id"]: person["name"] for person in members}
        all_works = get_team_works(connection, include_excluded=True)
        options = render_sidebar(members, labels)

        selected_works = scope_team_works(
            all_works,
            members,
            options["selected_members"],
            options["year_from"],
            options["year_to"],
            include_unassigned=options["include_unassigned"],
        )

        st.title("Paperazzi")
        st.caption("One team. Many papers. No duplicates. · OpenAlex, Crossref, and BibTeX")

        if st.session_state.get("report"):
            st.success(st.session_state.pop("report"))
        for warning in st.session_state.pop("sync_warnings", []):
            st.warning(warning)

        metrics = st.columns(6)
        metrics[0].metric("Selected members", len(options["selected_members"]))
        metrics[1].metric("Unique publications", len(selected_works))
        for index, category in enumerate(
            ("Articles", "Proceedings", "Book Chapters", "Patents"), start=2
        ):
            metrics[index].metric(category, len(category_works(selected_works, category)))

        if all_works and not selected_works:
            st.info(
                f"{len(all_works)} publications are stored in SQLite, but "
                "none match the current filters. Try All years, select more "
                "members, or include unassigned publications in the sidebar."
            )

        # Keyed tabs remember the active section across full-script reruns.
        # This also prevents a completed merge from sending the user back to Team.
        pages = st.tabs(
            [
                "👥 Team", "⬇️ Import", "📚 Library", "🔀 Reconcile",
                "📊 Statistics", "📦 Export",
            ],
            key="paperazzi_main_tab",
            on_change="rerun",
        )

        # Render only the active section. Hidden tabs do not need to query the
        # database or construct interactive widgets during another workflow.
        if pages[0].open:
            with pages[0]:
                render_team_tab(connection, members, labels)
        if pages[1].open:
            with pages[1]:
                render_import_tab(connection, members, labels, options)
        if pages[2].open:
            with pages[2]:
                render_library_tab(connection, all_works, members, labels, options)
        if pages[3].open:
            with pages[3]:
                render_reconcile_tab(connection, selected_works, members, options)
        if pages[4].open:
            with pages[4]:
                render_statistics_tab(
                    connection, selected_works, options["selected_members"], labels,
                    options["year_from"]
                )
        if pages[5].open:
            with pages[5]:
                render_export_tab(connection, selected_works, members)

        st.divider()
        st.caption("Paperazzi · local app · SQLite storage · MIT license")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
