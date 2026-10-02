# Metadata closure: first 200 remaining records

Reviewed 2026-10-02. This is the first 200 entries of `CATALOG-METADATA-ONLY.json`, from **N415 through C406**. Root owns the remaining 102 entries. The selection is disjoint from the agents' remaining-description audit.

`CATALOG-METADATA-CHAPTER.json` records all 200 decisions, retained families, source locators and hashes, proposed values and prior values. `CATALOG-METADATA-CHAPTER-PATCHES.json` contains the 142 proposed enrichment rows for the central CSV writer: 140 affiliation fields and five code/project fields, with overlap. There are no proposed changes to family, title, date or descriptions.

These proposals are **optional metadata enrichment**, not 142 factual errors. CONTRIBUTING permits an author-et-al fallback, which most proposed affiliation lists replace. Existing correct short institutional lists may also be expanded. Primary author/affiliation blocks were inspected for all records; N399 and N076 required fresh primary PDFs, and several HTML extractions required PDF footnotes or official project pages. Author-only fallback labels are retained where no institution was established. The global title/v1-date check and prior complete description review are reused, rather than presented as new independent reviews.

Editorial families were checked against each paper's mechanism and the site's definitions. Overlapping methods retain their existing reasonable primary family. For example, MAGMA-GEN's privileged coach produces corrections without per-step human demonstrations; `hitl` explicitly includes machine intervention, so the family is retained.

Nonempty code/project links were checked for an association with the paper: a primary text/href, a linked official project that names the repository, or a repository README/title linking the exact paper. Supplemental source chains and hashes record the cases missing from plain-text extraction. Blank fields make no release-availability claim. Five proposed additions point to explicitly linked repositories or official projects. This was an association review, not an installation, dependency-security or implementation-completeness audit of external repositories.

One availability nuance is retained in the evidence: N011's official project disables its Code button and labels it forthcoming, while embedding the existing repository URL; that repository identifies the same paper. Its association is verified without asserting a complete runnable release. CARF and the Wuji post-training additions are project pages, not claimed code releases.

Only review files were written. CSV and generated catalogue edits remain with the central writer, who can compare each proposed value against its recorded prior value before integration.
