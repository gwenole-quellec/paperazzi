# 📸 Paperazzi

**One team. Many papers. No duplicates.**

Paperazzi helps research teams **find, clean up, and showcase their publications**. It brings scattered references into one bibliography, removes duplicates, and turns the results into useful statistics and exportable publication lists.

It runs locally in your browser, with your data stored in SQLite on your computer.

## What you can do

- **Collect publications** from OpenAlex, Crossref, and BibTeX files (including Google Scholar and Zotero exports).
- **Build your team** by searching for researchers by name and selecting their OpenAlex profiles.
- **Curate your records**: find duplicates, correct author assignments and bibliographic details, and classify articles, proceedings, book chapters, datasets, and patents.
- **Explore and export** publication and citation statistics by year or researcher, with CSV and BibTeX exports ready for team reports.

## Get started

Requires **Python 3.10+**, an [OpenAlex API key](https://openalex.org/settings/api), and Python virtual-environment support.

On Ubuntu or Linux:

```bash
git clone https://github.com/gwenole-quellec/paperazzi.git
cd paperazzi
./start.sh
```

Open **http://localhost:8501**. Dependencies are installed automatically on first launch.

1. **Team:** Add researchers by name and confirm their OpenAlex profiles.
2. **Import:** Enter your API key in the sidebar, then import their publications. Add Crossref or BibTeX records as needed.
3. **Library & Reconcile:** Review publication types, correct records, and merge duplicates.
4. **Statistics & Export:** Explore the results and download your bibliography.

## Designed for team reporting

A coauthored paper counts **once for the team**, but contributes to each eligible researcher's individual statistics. Optional publication-year limits for each member help account for arrivals and departures. You can change the reporting period without deleting records.

Citation metrics depend on the coverage of external databases and should be checked before formal use.

## Local and open source

Your bibliography is stored in `data/paperazzi.sqlite`. Back it up by stopping Paperazzi and copying that file. Never commit your database or API key to a public repository.

**License:** [MIT](LICENSE).
