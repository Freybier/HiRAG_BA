# ADONIS JSONL-to-Text Converter for HiRAG (parent-only)

Deterministic, rule-based preprocessing tool that converts the ADONIS 40% JSONL
exports into clear, LLM-friendly textual records for HiRAG's entity/triplet
extraction. No LLM calls, embeddings, semantic similarity or API access are used
during conversion. The original JSONL files are never modified.

This package is independent of HiRAG and of the old converter
(`src/test-data/test-data_40pct/jsonl_to_text_converter_old.py`, which is left
untouched).

## Discovered input schemas (40% sample)

### `models_40pct.jsonl` (567 rows, all keys present on every row)

| Key                                                                                                                                   | Meaning                                                                                                 | Used as                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `model_id`                                                                                                                            | Braced UUID, e.g. `{0026db9b-...}`; unique                                                              | authoritative process ID (verbatim)                                                                  |
| `name`                                                                                                                                | Process name; unique in the sample                                                                      | authoritative process name (verbatim)                                                                |
| `description`                                                                                                                         | Constant `MT_BUSINESS_PROCESS_DIAGRAM_BPMN_20` in **all** rows — the ADONIS metaName, **not** free text | not emitted as `PROCESS_DESCRIPTION` (counted in the report)                                         |
| `purpose`                                                                                                                             | Constant `Geschäftsprozessdiagramm`                                                                     | sidecar only                                                                                         |
| `json_string`                                                                                                                         | Nested JSON with `id`, `name`, `type`, `metaName`, `rest_links`, `artefactType`, `libraryObjectId`      | sidecar only (`type`/`metaName` are constant across the corpus and are deliberately not textualized) |
| `parent_model_id`, `hierarchy_path`, `hierarchy_depth`                                                                                | ADONIS **model-tree** containment (a different structure than `C_SUB_PROCESS`)                          | sidecar only (not textualized, to avoid inventing hierarchy relations)                               |
| `pkk`, `image_data`, `source_hash`, `embedded_hash`, `source_refreshed_at`, `last_embedded_at`, `embedding_status`, `embedding_error` | pipeline bookkeeping                                                                                    | sidecar only                                                                                         |

### `hierarchy_40pct.jsonl` (823 rows)

| Key                                           | Meaning                                                                                 |
| --------------------------------------------- | --------------------------------------------------------------------------------------- |
| `parent_model_id`                             | relation **source** (the process that owns the relation)                                |
| `child_model_id`                              | relation **target**                                                                     |
| `relation_type`                               | `C_SUB_PROCESS` (334 rows) or `C_CROSS_REFERENCE` (489 rows); no other values           |
| `parent_model_name`, `child_model_name`       | redundant names; compared against the models table (authoritative), never used for text |
| `id`, `depth`, `hierarchy_path`, `created_at` | provenance/sidecar only                                                                 |

Direction verification (`parent_model_id` = parent, `child_model_id` = child):
the models table's tree-parent agrees with the forward reading 132 times vs 44
times for the reversed reading (partial signal only, because model-tree
containment and subprocess relations are different structures); the column names
are the primary evidence. The check runs on every conversion and its counts land
in `conversion_report.json` → `direction_check`. In the real sample: 0 duplicate
relationship triples, 0 broken references, 1 trivial name mismatch (trailing
space, normalized away by value cleaning).

### `documents_40pct.jsonl` (587 rows)

| Key                                                                                                                              | Meaning                                             |
| -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------- |
| `id`                                                                                                                             | numeric database key                                | authoritative document ID |
| `name`                                                                                                                           | document title                                      |
| `content`                                                                                                                        | extracted text (287 – 853,631 chars; median ≈ 7.4k) |
| `file_type`                                                                                                                      | `docx` (254), `pdf` (244), `xlsx` (85), `pptx` (4)  |
| `document_object_id`                                                                                                             | ADONIS object UUID (unbraced)                       |
| `document_object_url`, `user_facing_url`, `num_tokens`, `summary`, hashes, timestamps, `content_available`, `api_fetch_required` | sidecar only                                        |

**Process–document links do not exist in the export** (confirmed by the package
README/manifest). The converter never creates, infers or guesses them; every
document record carries `linked_process: unknown` and
`relationship_provenance: unavailable` in its structured metadata.

### `description_hierarchy-addon.json` (7,819 rows; optional, `--descriptions`)

Filtered export of the `data_vector_store2` chunk table, covering exactly the
same 567 models (the sparse `id` column is the full table's auto-increment PK —
gaps such as the "missing" ids 1–16 are expected and harmless). Chunk mix:
`overview` 567, `compliance` 567, `lane` 1,736, `activity` 4,382, `other` 567.

| Key           | Meaning                                                                                        | Used as                                                                           |
| ------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `model_id`    | bare lowercase UUID (the models export uses **braced** UUIDs — join via `strip("{}").lower()`) | join key to the models table                                                      |
| `object_type` | chunk kind                                                                                     | only `overview` rows are used                                                     |
| `text`        | generated chunk text                                                                           | `Beschreibung / Zweck:` → real `PROCESS_DESCRIPTION` (until the next field label) |
| `id`          | chunk-table primary key                                                                        | **never** joined to `pkk` in the models table (unrelated surrogate keys)          |
| `metadata_`   | nested JSON string; its inner `model_id` is mangled (`model_{uuid` artifact)                   | ignored                                                                           |

This file supplies the real process descriptions the models export lacks. When
passed via `--descriptions`, each model's overview-chunk description is emitted
as `PROCESS_DESCRIPTION` — unless the models export itself carries a real
description (differs from the metaName), which always wins. Empty values
(`Kein Eintrag`, blank) are treated as missing. Every process record reports its
provenance in `metadata.description_source` (`models_export` /
`description_addon` / `none`), and the report's `inputs.descriptions` section
audits matched/unmatched/empty rows.

### `chunks.jsonl` activity chunks (optional, `--include-activities`)

`chunks.jsonl` (built by `data-prep/convert_40pct_to_chunks.py`) carries one
`activity` chunk per Prozessschritt of a process model (4,382 chunks across 560
models in the 40% sample). Each chunk's text starts with
`Prozessschritt / Aktivität: <name>` and its metadata carries the numeric
`step_number`. With `--include-activities`, each process record gets an
`ACTIVITIES` section **after** the `SUBPROCESSES`/`CROSS_REFERENCES` sections,
ordered by step:

```text
ACTIVITIES:
Process "{PARENT_NAME}" has activity "{ACTIVITY_NAME}" (step 1).
```

Join rule: the chunk metadata's `model_id` (`model_{uuid}`, tolerating the
mangled `model_{uuid` artifact) is normalized to the bare lowercase UUID, same
as the models export's braced UUIDs. Both chunk layouts are accepted: the
production-shaped `chunks.jsonl` (fields nested under `metadata_`) and the raw
chunk export (top-level fields). Activity chunks for models outside the models
export are reported under `inputs.activities.unmatched_model_ids`, never
textualized; every process record reports `metadata.activity_count`. Note: like
`--descriptions`, `--limit` is NOT applied to the chunks file. The file names
are unchanged — when the flag is passed to `data-prep/data_preparation.py`
(`--mode activities`), the corpus is written to the separate
`hirag_parent_only_output_with_activities/` directory so both variants coexist.
Enabling activities changes the record texts and therefore HiRAG's content
hashes (re-index into a fresh working dir).

### `chunks.jsonl` lane/compliance/other chunks (optional, `--include-everything`)

`--include-everything [CHUNKS_JSONL]` (the `everything` mode of
`data-prep/data_preparation.py`) implies `--include-activities` and additionally
appends the remaining model chunk types of the chunks corpus as sections
**after** `ACTIVITIES`:

- `lane` chunks (1,736 across 522 models) → `LANES:` sentences; the lane name
  comes from the `lane_name` metadata (fallback: the `Lane-Name:` text line),
  the swimlane/pool distinction from `object_meta_name` (fallback: `Typ:`), and
  the number from the `_lane_<n>` suffix of the chunk's `node_id`:
  `Process "{PARENT_NAME}" has swimlane "{LANE_NAME}" (Nr. 0).` /
  `... has pool "{POOL_NAME}" (Nr. 1).`
- `compliance` chunks (567, one per model) → `COMPLIANCE:` sentences from the
  `status` / `risk_level` metadata (fallback: `Status:` / `Risikostufe:` text
  lines): `Process "{PARENT_NAME}" has the status "{STATUS}".` and
  `... has the risk level "{RISK_LEVEL}".` Empty values and the `Kein Eintrag`
  placeholder are treated as missing (in the current corpus every `risk_level`
  is `Kein Eintrag`, so only the status sentence renders).
- `other` chunks (567, one per model) → `RELATED_OBJECTS:` sentences from the
  verbatim `Weitere Objekte in:` listing lines: the relation tag
  `[C_CROSS_REFERENCE]` / `[C_SUB_PROCESS]` keeps the
  `includes ... as cross-reference/subprocess` wording, every other tag renders
  as `Process "{PARENT_NAME}" has "{NAME}" as {TAG_WITHOUT_C_PREFIX}.` (e.g.
  `START_EVENT`), untagged lines as `... as related object.` Note these listings
  include out-of-sample process references (the chunks corpus keeps the full
  verbatim relation listings), unlike the verified
  `SUBPROCESSES`/`CROSS_REFERENCES` sections.

Chunk header lines and the trailing `Quelle:` source-URL line are never
textualized. Unmatched/empty rows are audited under `inputs.activities.extras`
(`unmatched_model_ids`, `empty_rows`, per-type matches); records report
`metadata.lane_count`, `metadata.has_compliance` and
`metadata.related_object_count`. When the mode is passed to
`data-prep/data_preparation.py`, the corpus is written to the separate
`hirag_parent_only_output_with_everything/` directory.

## Assumptions that could affect correctness

1. Relationship direction follows the column names (`parent_model_id` →
   source/parent, `child_model_id` → target/child), corroborated by the
   tree-parent cross-check. If a future export encodes direction differently,
   the `direction_check` report section will flag it (verdict starts with
   `ambiguous`) — it is never silently flipped.
2. `models.description` containing only the metaName constant is treated as "no
   description". If a future export stores real descriptions there, they are
   emitted as `PROCESS_DESCRIPTION` whenever they differ from the metaName.
3. The 40% sample is incomplete by design: absence of an outgoing relationship
   is never stated in the text (no "has no subprocesses" claims), and no
   inverse/parent statements are generated to compensate.
4. Chunking is character-based, not token-based (~4 chars/token heuristic
   motivates the 4000-char default ≈ HiRAG's 1200-token chunk size).

## Install / requirements

Python 3.12+ standard library only. No dependencies.

## Usage

```bash
cd src/test-data/data-prep/adonis_converter
python converter.py \
  --models ../../test-data_40pct/models_40pct.jsonl \
  --hierarchy ../../test-data_40pct/hierarchy_40pct.jsonl \
  --documents ../../test-data_40pct/documents_40pct.jsonl \
  --output ./output \
  --hierarchy-format parent-only
```

Also runnable as a module from the `data-prep` directory:
`python -m adonis_converter.converter ...`

Options: `--chunk-documents` (split large documents into sections; \*\*off by

default** — HiRAG chunks internally, so each document is emitted as one whole
record), `--max-section-chars` (default 4000), `--section-overlap-chars`
(default 200), `--process-warn-chars` (default 8000), `--export-txt` (one plain
`.txt` per record under `output/txt/`), `--combined-text` (see below),
`--descriptions PATH` (optional `data_vector_store2` chunk export, JSON array or
JSONL, supplying real process descriptions — see above),
`--include-activities [CHUNKS_JSONL]` (append each process's activity steps as
an `ACTIVITIES` section after the relationship sections; optionally takes the
chunks corpus path, default `chunks.jsonl` next to the models export; off by
default), `--include-everything [CHUNKS_JSONL]` (like `--include-activities`,
plus `LANES`/`COMPLIANCE`/`RELATED_OBJECTS` sections from the lane/compliance/
other chunks — see above), `--exclude-file-type EXT` (repeatable,
case-insensitive; nothing is excluded **by default\*\*, each flag adds a type;
excluded documents are skipped entirely and listed under
`documents.excluded_by_file_type` in the report), `--limit N` (smoke tests; not
applied to the descriptions/activities chunk files, whose chunks are grouped per
model). `--hierarchy-format` accepts only `parent-only`; bidirectional
textualization is intentionally not implemented.

## Combined array-ready text files

With `--combined-text`, two additional files are written:

- `processes_text.txt` — all process record texts in one file
- `documents_text.txt` — all document/section record texts in one file

Records are separated by a deterministic delimiter line
(`=== RECORD SEPARATOR ===`), so each file splits into an array with one element
per record:

```python
texts = path.read_text(encoding="utf-8").split("\n=== RECORD SEPARATOR ===\n")
hirag.insert(texts)  # HiRAG accepts a list of strings
```

The delimiter is guaranteed never to appear inside a record text (the converter
hard-fails if it ever would). The combined files contain plain record text only
— no JSON serialization syntax.

## Outputs (in `--output` directory)

| File                                                   | Content                                                                                                                                                      |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `processes_textualized.jsonl`                          | one record per process; verified outgoing subprocesses/cross-references embedded in the source record only                                                   |
| `documents_textualized.jsonl`                          | one `DOCUMENT` record per document (full content; `DOCUMENT_SECTION` records only with `--chunk-documents`)                                                  |
| `relationship_ground_truth.jsonl`                      | all verified, deduplicated `(source_id, relationship_type, target_id)` edges with names and provenance                                                       |
| `metadata_mapping.jsonl`                               | generated record → original row (document `content` omitted, reconstructable via `source_file`/`source_line`; plus section offsets when chunking is enabled) |
| `entity_lookup.json`                                   | authoritative ID → name lookup for processes and documents                                                                                                   |
| `conversion_report.json`                               | statistics, direction check, schema mapping, audit lists, warnings                                                                                           |
| `txt/` (optional)                                      | one plain-text file per record, no JSON syntax                                                                                                               |
| `processes_text.txt` / `documents_text.txt` (optional) | one combined plain-text file each, records separated by `=== RECORD SEPARATOR ===` lines for easy array splitting                                            |

## Parent-only rule

Each verified relationship is textualized exactly once, in the source (parent)
process's record:

```text
SUBPROCESSES:
Process "{PARENT_NAME}" has subprocess "{CHILD_NAME}".
```

Child records never repeat that they are a subprocess; no inverse statements, no
recursive grandchildren. The validator hard-fails if any relationship is
missing, duplicated, leaked into an unrelated record, or repeated in the child's
record.

## Record text format

The record text carries only the per-record semantics; everything else lives in
the JSONL `metadata` / sidecar files:

- Processes: `PROCESS_NAME`, optional `PROCESS_DESCRIPTION`, optional
  `SUBPROCESSES` / `CROSS_REFERENCES` (names-only sentences — process IDs are
  kept in the JSONL `source_id`/metadata, not in the text), optional
  `ACTIVITIES` (only with `--include-activities`; step-ordered names-only
  sentences after the relationship sections), optional `LANES` / `COMPLIANCE` /
  `RELATED_OBJECTS` (only with `--include-everything`, after `ACTIVITIES`).
  `metaName`/`type` are deliberately excluded because they collapse to one or
  two constant values across the corpus and do not differentiate processes.
- Documents: `DOCUMENT_TITLE` + `DOCUMENT_CONTENT` only (plus a `SECTION: n/t`
  line when `--chunk-documents` is enabled). IDs and file type stay in the
  metadata.

## Validation behavior

- Malformed JSONL, duplicate process/document IDs, missing process records,
  parent-only violations, section-coverage gaps → explicit failure (exit code
  2), never silent.
- Missing optional fields, rejected rows, unknown relation types, broken
  references, duplicate source rows, name mismatches → reported in
  `conversion_report.json`, never silently dropped.
- When `--chunk-documents` is enabled, document sections record
  `content_start`/`content_end` offsets into the original content plus
  `overlap_with_previous_chars`; overlap is repeated context, not independent
  source information.

## HiRAG insertion example

```bash
python hirag_loader_example.py --output ./output            # dry run
python hirag_loader_example.py --output ./output --insert   # real insert
```

`HiRAG.insert()` accepts a string or a list of strings; each string gets its own
HiRAG-internal hash. The loader passes only the plain `text` fields (never JSON
syntax) in deterministic batches. Note that HiRAG applies its own internal
chunking on top — one record does not guarantee one HiRAG chunk. Real insertion
requires a configured HiRAG instance and LLM credentials; conversion and tests
never call HiRAG or an LLM.

## Tests

```bash
cd src/test-data && ../.venv/bin/python -m pytest data-prep/adonis_converter/tests -q
```

46 tests cover: simple parent-child, three-level hierarchy, leaf processes,
multiple children, cross-references, combined relations, relation-less
processes, broken references, duplicate rows, identical names with different
IDs, no-chunking default for large documents, opt-in long-document splitting +
exact reconstruction, unlinked documents, missing attributes, German special
characters, malformed/empty JSONL, duplicate-ID hard failure, unknown relation
types, combined-text export (array split round-trip), file-type exclusion
(explicit opt-in only, nothing excluded by default), byte-level output
determinism, activities inclusion (section order after the relationship
sections, step ordering, model-ID normalization incl. the mangled `model_{uuid`
artifact and nested `metadata_` shape, unmatched models reported, nameless/
stepless chunks, missing chunks file, off by default), extra chunk types
(everything mode: section order after ACTIVITIES, swimlane/pool + Nr. sentences,
compliance status/risk-level sentences with `Kein Eintrag` skipped,
related-object tag rendering, unmatched models reported, config validation, off
by default), and a smoke test against the real sample.

## Structure

```text
adonis_converter/
    config.py                 # configuration + explicit field mappings
    schema.py                 # JSONL reading, NULL handling, typed records
    relationship_mapper.py    # canonical dedup, direction check, audit
    process_textualizer.py    # parent-only process records
    document_textualizer.py   # document records + deterministic chunking
    description_addon.py      # overview-chunk descriptions addon
    activity_addon.py         # activity steps + lane/compliance/other extras (chunks.jsonl)
    validator.py              # input/output integrity checks
    converter.py              # orchestrator + CLI
    hirag_loader_example.py   # HiRAG insertion example
    tests/                    # synthetic-dataset pytest suite
    README.md
```
