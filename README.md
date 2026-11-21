# 📸 Paperazzi

**One team. Many papers. No duplicates.**

Paperazzi is a local web application for collecting, curating, and analyzing the research output of a team. It consolidates records from multiple bibliographic sources, identifies duplicates, keeps track of individual contributors, and generates publication lists and bibliometric summaries for reporting.

Built with **Python**, **Streamlit**, and **SQLite**, Paperazzi runs in your browser while retaining the bibliography on your own computer. It does not require a hosted service or a separate database server.

## Features

- **Team roster:** Register researchers using one or more OpenAlex Author IDs and set optional publication-year limits for each member.
- **Bibliographic imports:** Synchronize [OpenAlex](https://openalex.org/) author profiles, search [Crossref](https://www.crossref.org/), or import `.bib` files from Google Scholar, Zotero, HAL, and other reference managers.
- **Deduplicated library:** Combine records describing the same work, preserve the original sources, and manually correct remaining mismatches.
- **Publication categories:** Browse **Articles**, **Proceedings**, **Book Chapters**, **Patents**, and **Other** works, in addition to a consolidated **All** view.
- **Editable metadata:** Correct titles, years, BibTeX entry types, and team-member assignments; exclude records from reports without deleting them.
- **Team statistics:** Explore unique publications by year and entry type, publications and indexed citations by researcher, available citation counts, h-index, and i10-index.
- **Exports:** Download the scoped bibliography in CSV or BibTeX format, category-specific CSV files, a team roster, or a detailed JSON reference archive.

## Requirements

- Python **3.10 or later** (3.11 or 3.12 recommended)
- Internet access for installing dependencies and using online bibliographic APIs
- **Streamlit 1.55 or later**, installed through `requirements.txt`
- An **OpenAlex API key** for OpenAlex synchronization

## Installation

On Ubuntu, install the virtual-environment tools if necessary:

```bash
sudo apt install python3-venv python3-pip
```

In the extracted project directory, run:

```bash
cd paperazzi
./start.sh
```

The launcher creates a `.venv` directory and installs the required packages on first use. When upgrading an existing installation, it also upgrades an outdated Streamlit version so that tab selection survives a refresh. Open **http://localhost:8501** in a browser.

Alternatively, create a virtual environment manually and start Streamlit:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

### OpenAlex API key

Enter the key in the sidebar, or set an environment variable before starting the application:

```bash
export OPENALEX_API_KEY='your_openalex_api_key'
./start.sh
```

An optional contact email can be provided through `PAPERAZZI_EMAIL` or the sidebar. The API key is not stored in SQLite; do not commit it to Git.

## Getting started

1. **Add team members.** Open **Team**, enter first and last names, and associate one or more OpenAlex Author IDs with each researcher. To change an existing member, **click their row** in the team table and edit the form below. A new database starts with an empty roster.
2. **Choose your reporting period.** Set the team-wide publication-year range in the sidebar. In **Team**, optionally configure individual start and end years; both are inclusive.
3. **Synchronize OpenAlex.** In **Import**, choose the researchers to synchronize and run the import. Each member's individual date limits apply to the OpenAlex request. Applying the team-wide period during import is optional.
4. **Review the library.** Use **Library** to inspect publication categories and team attribution. **Click a publication row** to open its correction form below. This lets you edit its title, publication year, BibTeX type, and team-member attribution, or exclude it from the report.
5. **Resolve duplicates.** In **Reconcile**, inspect fuzzy title matches and manually merge or split source records. After confirming a merge or split, Paperazzi stays on the Reconcile tab, refreshes the candidate list using the same similarity threshold, and displays the next pair.
6. **Fill gaps.** Search Crossref and review matching results, or import one or more BibTeX files. Bibliographies containing works from different authors can be imported without assigning every record to a single researcher.
7. **Explore and export.** Inspect **Statistics** and download the appropriate files from **Export** or directly from a Library category.

Imports are incremental. Re-importing an existing source record updates its metadata instead of creating another copy. Changes are reflected in the interface after the import finishes. The active main tab is preserved after database updates.

## Publication categories

Paperazzi preserves the effective **BibTeX entry type** of each work. This is either inferred from the source, provided by an imported BibTeX record, or manually overridden in the Library interface.

| Library category | BibTeX entry type(s) | Typical content |
|---|---|---|
| Articles | `article` | Journal articles |
| Proceedings | `inproceedings`, `conference` | Papers published in conference proceedings |
| Book Chapters | `incollection` | Chapters in edited books or collections |
| Patents | `patent` | Patent records |
| Other | All remaining types | Books, theses, reports, preprints, and other records |

An `@incollection` BibTeX entry is retained as a book chapter during import and export. Crossref's `book-chapter` and OpenAlex works identified as book chapters are also mapped to `incollection`. The **Book Chapters** category has its own listing, summary counter, and CSV export.

Bibliographic databases occasionally misclassify proceedings and chapters; manually changing the entry type updates the category without destroying the source metadata.

## How publications are counted

A single publication coauthored by several team members is stored **once** in the collective bibliography, but can count toward each eligible member's individual total.

Individual start and end years refer strictly to the **publication year**: they do not validate institutional affiliation or employment dates. For example, a member configured for 2024–2026 can be credited for a 2024 paper, but not a 2023 paper. A paper counts in the team total if at least one selected member is eligible for its publication year.

The sum of individual contributions may therefore exceed the number of unique team publications. Citation totals per member are calculated by crediting each eligible coauthor with the full current citation count of each of their publications; they are not divided among coauthors. As a result, the sum of individual citation totals can also exceed the team total. The Statistics page displays publication types and annual publication counts in the first chart row, followed by publications and citations by member in the second row. Former members can remain part of a historical reporting period regardless of their current active/inactive status.

The global year range filters the Library, Statistics, and bibliographic exports. Individual limits apply to OpenAlex synchronization; an option can additionally limit import to the global range. Changing the reporting period never deletes already stored records.

## Source data and reconciliation

- **OpenAlex** is the primary automatic source. Team members are identified by their OpenAlex Author IDs (such as `A123456789`). A single person may be associated with multiple IDs if OpenAlex split their profile.
- **Crossref** provides an optional, manually reviewed search by author name. Name searches can include namesakes or miss publications with complex author names.
- **BibTeX** can supplement missing references and preserve explicitly specified entry types, including `@incollection` and `@patent`.

Matching uses identical DOIs, OpenAlex Work IDs, or strictly normalized titles with matching publication years. Similar titles are **suggestions**, not automatic merges. Each consolidated work retains its individual source records so that decisions can be reviewed later.

The citation count is selected from one available source according to a fixed priority; counts from different sources are **not summed**. Bibliometric indicators depend on source coverage and should be reviewed before inclusion in an official report.

### Known limitations

- OpenAlex is not an exhaustive patent database. Importing `@patent` BibTeX entries is a practical way to complete a patent inventory; no patent-office connector is provided.
- Conference proceedings and book chapters may be misclassified by upstream services; corrections are available in the Library.
- Importing a mixed BibTeX bibliography with several members selected assigns **all** of its entries to **every** selected member. Leave the assignment empty for mixed files and verify records individually.
- A publication without a known year is omitted when bounded year filters apply.
- The JSON export is a detailed reference archive and **cannot yet be restored** directly through the interface.

## Local data and backups

The default SQLite database is stored at:

```text
data/paperazzi.sqlite
```

Set `PAPERAZZI_DB` to use another path. The `data/` path is excluded from version control by `.gitignore`.

For a restorable backup, **stop Paperazzi** and copy the SQLite file to a safe location:

```bash
cp data/paperazzi.sqlite data/paperazzi.sqlite.bak
```

Existing team and publication data remain available when updating the application files. Avoid distributing personal team databases or API keys with the source code.

## Project structure

```text
paperazzi/
├── app.py                  # Streamlit interface and user workflows
├── paperazzi/
│   ├── core.py             # Canonical works, source records, and curation
│   ├── sources.py          # OpenAlex, Crossref, and BibTeX adapters
│   └── team.py             # Team membership, attribution, and report scope
├── .streamlit/config.toml  # Streamlit configuration
├── start.sh
├── requirements.txt
├── LICENSE
└── README.md
```

The codebase separates presentation (`app.py`), canonical bibliography operations (`core.py`), external-source adapters (`sources.py`), and team-specific behavior (`team.py`). Public functions have docstrings explaining their responsibilities and key assumptions.

## License

MIT. See [LICENSE](LICENSE).
