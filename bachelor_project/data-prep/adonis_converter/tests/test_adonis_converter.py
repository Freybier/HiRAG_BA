"""Tests for the ADONIS JSONL-to-text converter.

All tests use small synthetic datasets; the real 40% sample is never
required. No LLM calls, embeddings or HiRAG indexing are executed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from adonis_converter.config import ConverterConfig
from adonis_converter.converter import COMBINED_RECORD_DELIMITER, run_conversion
from adonis_converter.schema import SchemaError
from adonis_converter.validator import ConversionError


def _model(model_id: str, name: str, **extra: Any) -> dict[str, Any]:
    """Build a minimal synthetic model row."""
    row = {"model_id": model_id, "name": name}
    row.update(extra)
    return row


def _relation(
    source_id: str,
    target_id: str,
    relation_type: str = "C_SUB_PROCESS",
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal synthetic hierarchy row."""
    row = {
        "parent_model_id": source_id,
        "child_model_id": target_id,
        "relation_type": relation_type,
    }
    row.update(extra)
    return row


def _document(doc_id: str, name: str, content: str, **extra: Any) -> dict[str, Any]:
    """Build a minimal synthetic document row."""
    row = {"id": doc_id, "name": name, "content": content}
    row.update(extra)
    return row


def _overview_chunk(
    model_id: str, description: str | None, **extra: Any
) -> dict[str, Any]:
    """Build a minimal synthetic overview chunk row for the addon file."""
    text = (
        "Prozessmodell: Beispiel\n"
        "Typ: MT_BUSINESS_PROCESS_DIAGRAM_BPMN_20 / DIAGRAM\n"
        f"Beschreibung / Zweck: {description if description is not None else 'Kein Eintrag'}\n"
        "Verantwortlich: Max Mustermann\n"
        "Quelle: https://example.invalid/view\n"
    )
    row = {
        "id": 17,
        "node_id": f"model_{model_id}_overview",
        "model_id": model_id,
        "object_type": "overview",
        "text": text,
    }
    row.update(extra)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write synthetic rows as JSONL."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _activity_chunk(
    model_id: str,
    name: str | None,
    step: int | None,
    nested: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal synthetic activity chunk row (chunks.jsonl shape).

    Args:
        model_id: Model reference; with ``nested=True`` the production
            ``model_{uuid}`` metadata form is used, otherwise the raw
            top-level addon form.
        name: Activity name; ``None`` omits the label from the text.
        step: Step number; ``None`` renders an empty ``step_number``.
        nested: Use the nested ``metadata_`` production shape.
        extra: Extra row fields.
    """
    first_line = f"Prozessschritt / Aktivität: {name}" if name is not None else "Notiz:"
    text = (
        f"{first_line}\n"
        f"Reihenfolge (Schritt {step if step is not None else ''})\n"
        "Typ: C_TASK\n"
        "Prozessmodell: Beispiel\n"
        "Modell-Status: Gültig\n"
        "Quelle: https://example.invalid/view\n"
    )
    if nested:
        row: dict[str, Any] = {
            "id": 22,
            "text": text,
            "metadata_": {
                "model_id": f"model_{{{model_id}}}",
                "object_type": "activity",
                "step_number": "" if step is None else str(step),
            },
        }
    else:
        row = {
            "id": 22,
            "model_id": model_id,
            "object_type": "activity",
            "step_number": "" if step is None else str(step),
            "text": text,
        }
    row.update(extra)
    return row


def _run(
    tmp_path: Path,
    models: list[dict[str, Any]],
    hierarchy: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    descriptions: list[dict[str, Any]] | None = None,
    activities: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Run the converter on synthetic inputs and return parsed outputs."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    models_path = tmp_path / "models.jsonl"
    hierarchy_path = tmp_path / "hierarchy.jsonl"
    documents_path = tmp_path / "documents.jsonl"
    _write_jsonl(models_path, models)
    _write_jsonl(hierarchy_path, hierarchy)
    _write_jsonl(documents_path, documents)
    if descriptions is not None:
        # The real addon is a single JSON array, not JSONL.
        descriptions_path = tmp_path / "descriptions.json"
        descriptions_path.write_text(
            json.dumps(descriptions, ensure_ascii=False), encoding="utf-8"
        )
        overrides.setdefault("descriptions_path", descriptions_path)
    if activities is not None:
        # The real chunks corpus is newline-delimited JSON.
        activities_path = tmp_path / "chunks.jsonl"
        _write_jsonl(activities_path, activities)
        overrides.setdefault("activities_path", activities_path)
    config = ConverterConfig(
        models_path=models_path,
        hierarchy_path=hierarchy_path,
        documents_path=documents_path,
        output_dir=tmp_path / "output",
        **overrides,
    )
    report = run_conversion(config)
    processes = [
        json.loads(line)
        for line in (config.output_dir / config.processes_file)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    docs = [
        json.loads(line)
        for line in (config.output_dir / config.documents_file)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    ground_truth = [
        json.loads(line)
        for line in (config.output_dir / config.ground_truth_file)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return report, processes, docs, ground_truth


def _sub_sentence(parent_name: str, child_name: str) -> str:
    """Render the expected subprocess sentence for assertions."""
    return f'Process "{parent_name}" has subprocess "{child_name}".'


def test_simple_parent_child(tmp_path: Path) -> None:
    """A single parent-child edge is textualized only in the parent."""
    _, processes, _, ground_truth = _run(
        tmp_path,
        models=[_model("m_x", "Process X"), _model("m_y", "Process Y")],
        hierarchy=[_relation("m_x", "m_y")],
        documents=[],
    )
    assert len(processes) == 2
    text_x = next(p for p in processes if p["source_id"] == "m_x")["text"]
    text_y = next(p for p in processes if p["source_id"] == "m_y")["text"]
    sentence = _sub_sentence("Process X", "Process Y")
    assert sentence in text_x
    assert "SUBPROCESSES" in text_x
    assert "SUBPROCESSES" not in text_y
    assert "subprocess" not in text_y
    assert len(ground_truth) == 1
    assert ground_truth[0]["relationship_type"] == "C_SUB_PROCESS"


def test_three_level_hierarchy_parent_only(tmp_path: Path) -> None:
    """X->Y->Z: Y mentions Z but never that it is a subprocess of X."""
    _, processes, _, ground_truth = _run(
        tmp_path,
        models=[
            _model("m_x", "Process X"),
            _model("m_y", "Process Y"),
            _model("m_z", "Process Z"),
        ],
        hierarchy=[
            _relation("m_x", "m_y"),
            _relation("m_y", "m_z"),
        ],
        documents=[],
    )
    texts = {p["source_id"]: p["text"] for p in processes}
    assert _sub_sentence("Process X", "Process Y") in texts["m_x"]
    assert _sub_sentence("Process Y", "Process Z") in texts["m_y"]
    # Parent-only rule: no inverse or repeated statements.
    assert "subprocess" not in texts["m_z"]
    assert _sub_sentence("Process X", "Process Y") not in texts["m_y"]
    assert _sub_sentence("Process Y", "Process Z") not in texts["m_z"]
    # No recursive grandchildren in X's record.
    assert "Process Z" not in texts["m_x"]
    # Both verified edges are in the ground truth.
    edges = {(g["source_id"], g["target_id"]) for g in ground_truth}
    assert edges == {("m_x", "m_y"), ("m_y", "m_z")}


def test_leaf_process_without_subprocesses(tmp_path: Path) -> None:
    """A leaf process gets a minimal record without a SUBPROCESSES section."""
    _, processes, _, _ = _run(
        tmp_path,
        models=[
            _model("m_x", "Process X"),
            _model("m_y", "Process Y"),
        ],
        hierarchy=[_relation("m_x", "m_y")],
        documents=[],
    )
    text_y = next(p for p in processes if p["source_id"] == "m_y")["text"]
    assert "PROCESS_NAME: Process Y" in text_y
    # IDs stay in the JSONL metadata, not in the text.
    assert "PROCESS_ID" not in text_y
    assert "RECORD_TYPE" not in text_y
    assert "SUBPROCESSES" not in text_y
    assert "CROSS_REFERENCES" not in text_y
    # No "has no subprocesses" claims: absence in a sample is not proof.
    assert "no subprocess" not in text_y


def test_process_with_multiple_children(tmp_path: Path) -> None:
    """All direct children appear exactly once in the parent record."""
    _, processes, _, ground_truth = _run(
        tmp_path,
        models=[
            _model("m_x", "Process X"),
            _model("m_a", "Process A"),
            _model("m_b", "Process B"),
            _model("m_c", "Process C"),
        ],
        hierarchy=[
            _relation("m_x", "m_b"),
            _relation("m_x", "m_a"),
            _relation("m_x", "m_c"),
        ],
        documents=[],
    )
    text_x = next(p for p in processes if p["source_id"] == "m_x")["text"]
    for child_name in ("Process A", "Process B", "Process C"):
        sentence = _sub_sentence("Process X", child_name)
        assert text_x.count(sentence) == 1
    # Deterministic alphabetical ordering by child name.
    assert (
        text_x.index("Process A")
        < text_x.index("Process B")
        < text_x.index("Process C")
    )
    assert len(ground_truth) == 3


def test_process_with_cross_references(tmp_path: Path) -> None:
    """Cross-references are attached to the source process only."""
    _, processes, _, ground_truth = _run(
        tmp_path,
        models=[_model("m_a", "Process A"), _model("m_b", "Process B")],
        hierarchy=[_relation("m_a", "m_b", "C_CROSS_REFERENCE")],
        documents=[],
    )
    text_a = next(p for p in processes if p["source_id"] == "m_a")["text"]
    text_b = next(p for p in processes if p["source_id"] == "m_b")["text"]
    sentence = 'Process "Process A" cross-references "Process B".'
    assert "CROSS_REFERENCES" in text_a
    assert sentence in text_a
    assert "CROSS_REFERENCES" not in text_b
    assert "cross-referenced by" not in text_b
    assert ground_truth[0]["relationship_type"] == "C_CROSS_REFERENCE"


def test_process_with_children_and_cross_references(tmp_path: Path) -> None:
    """Subprocesses and cross-references are separate, ordered sections."""
    _, processes, _, _ = _run(
        tmp_path,
        models=[
            _model("m_x", "Process X"),
            _model("m_y", "Process Y"),
            _model("m_z", "Process Z"),
        ],
        hierarchy=[
            _relation("m_x", "m_z", "C_CROSS_REFERENCE"),
            _relation("m_x", "m_y", "C_SUB_PROCESS"),
        ],
        documents=[],
    )
    text_x = next(p for p in processes if p["source_id"] == "m_x")["text"]
    assert "SUBPROCESSES" in text_x
    assert "CROSS_REFERENCES" in text_x
    assert text_x.index("SUBPROCESSES") < text_x.index("CROSS_REFERENCES")
    assert _sub_sentence("Process X", "Process Y") in text_x
    assert 'cross-references "Process Z".' in text_x


def test_process_with_no_outgoing_relationships(tmp_path: Path) -> None:
    """A process without outgoing relations has no RELATIONSHIPS block."""
    _, processes, _, ground_truth = _run(
        tmp_path,
        models=[_model("m_a", "Solo Process")],
        hierarchy=[],
        documents=[],
    )
    text = processes[0]["text"]
    assert "RELATIONSHIPS" not in text
    assert ground_truth == []


def test_relationship_to_nonexistent_process(tmp_path: Path) -> None:
    """Broken references are reported, never textualized or invented."""
    report, processes, _, ground_truth = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[_relation("m_x", "m_missing"), _relation("m_ghost", "m_x")],
        documents=[],
    )
    assert ground_truth == []
    assert len(report["relationships"]["broken_references"]) == 2
    text_x = processes[0]["text"]
    assert "RELATIONSHIPS" not in text_x
    assert "m_missing" not in text_x


def test_duplicate_source_relationships(tmp_path: Path) -> None:
    """Duplicate rows are textualized once and kept in the audit data."""
    report, processes, _, ground_truth = _run(
        tmp_path,
        models=[_model("m_x", "Process X"), _model("m_y", "Process Y")],
        hierarchy=[
            _relation("m_x", "m_y"),
            _relation("m_x", "m_y"),
            _relation("m_x", "m_y"),
        ],
        documents=[],
    )
    text_x = next(p for p in processes if p["source_id"] == "m_x")["text"]
    sentence = _sub_sentence("Process X", "Process Y")
    assert text_x.count(sentence) == 1
    assert len(ground_truth) == 1
    assert ground_truth[0]["occurrences_in_source"] == 3
    assert report["relationships"]["duplicates_removed"] == 2
    assert len(report["relationships"]["duplicate_details"]) == 2


def test_identical_names_different_ids(tmp_path: Path) -> None:
    """Same-named processes keep distinct records via their source IDs."""
    _, processes, _, _ = _run(
        tmp_path,
        models=[
            _model("m_1", "Vorgang bearbeiten"),
            _model("m_2", "Vorgang bearbeiten"),
        ],
        hierarchy=[_relation("m_1", "m_2")],
        documents=[],
    )
    text_1 = next(p for p in processes if p["source_id"] == "m_1")["text"]
    sentence = _sub_sentence("Vorgang bearbeiten", "Vorgang bearbeiten")
    assert sentence in text_1
    assert sentence.count("Vorgang bearbeiten") == 2
    # The text is name-only, so the records stay distinguishable through
    # their JSONL metadata (source_id / record_id), not the text.
    assert {p["source_id"] for p in processes} == {"m_1", "m_2"}


def test_long_document_section_splitting(tmp_path: Path) -> None:
    """Long documents split deterministically and reconstruct exactly.

    Source-side chunking is opt-in (HiRAG chunks internally), so the test
    enables it explicitly.
    """
    paragraphs = [
        f"Paragraph {index} enthält eine vollständige Anweisung." for index in range(40)
    ]
    content = "\n\n".join(paragraphs)
    _, _, docs, _ = _run(
        tmp_path,
        models=[],
        hierarchy=[],
        documents=[_document("d1", "Lang Richtlinie", content)],
        chunk_documents=True,
        max_section_chars=200,
        section_overlap_chars=40,
    )
    assert len(docs) > 1
    assert all(rec["record_type"] == "DOCUMENT_SECTION" for rec in docs)
    numbers = [rec["metadata"]["section_number"] for rec in docs]
    assert numbers == list(range(1, len(docs) + 1))
    assert all(
        f"doc_d1_section_{n:04d}" == rec["metadata"]["section_id"]
        for rec, n in zip(docs, numbers)
    )
    # Every section preserves the document title and its position.
    assert all("DOCUMENT_TITLE: Lang Richtlinie" in rec["text"] for rec in docs)
    assert all(
        f"SECTION: {n}/{len(docs)}" in rec["text"] for rec, n in zip(docs, numbers)
    )
    # IDs stay in the JSONL metadata, not in the text.
    assert all("DOCUMENT_ID" not in rec["text"] for rec in docs)
    # Exact reconstruction accounting for the recorded overlap.
    rebuilt = ""
    for rec in docs:
        meta = rec["metadata"]
        start = int(meta["content_start"]) + int(meta["overlap_with_previous_chars"])
        rebuilt += content[start : int(meta["content_end"])]
    assert rebuilt == content
    # The overlapped text is flagged as repetition, not independent content.
    assert int(docs[1]["metadata"]["overlap_with_previous_chars"]) > 0


def test_documents_not_chunked_by_default(tmp_path: Path) -> None:
    """Without --chunk-documents, even huge documents stay one record."""
    content = "\n\n".join(f"Absatz {index} mit etwas Inhalt." for index in range(500))
    report, _, docs, _ = _run(
        tmp_path,
        models=[],
        hierarchy=[],
        documents=[_document("d1", "Sehr langes Dokument", content)],
    )
    assert len(docs) == 1
    assert docs[0]["record_type"] == "DOCUMENT"
    assert docs[0]["record_id"] == "document_doc_d1"
    assert content in docs[0]["text"]
    assert "SECTION" not in docs[0]["text"]
    assert report["documents"]["chunked_documents"] == 0


def test_document_without_process_relationship(tmp_path: Path) -> None:
    """Documents never claim process links; limitation metadata is set."""
    _, _, docs, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[_document("d1", "Richtlinie", "Kurzer Inhalt.")],
    )
    assert len(docs) == 1
    record = docs[0]
    assert record["record_type"] == "DOCUMENT"
    assert record["metadata"]["linked_process"] == "unknown"
    assert record["metadata"]["relationship_provenance"] == "unavailable"
    assert "Process X" not in record["text"]


def test_missing_optional_attributes(tmp_path: Path) -> None:
    """Missing optional fields produce no empty attribute sections."""
    _, processes, docs, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[_document("d1", "Doc", "Inhalt")],
    )
    text = processes[0]["text"]
    assert "PROCESS_ATTRIBUTES" not in text
    assert "PROCESS_DESCRIPTION" not in text
    assert "ENTITY_TYPE" not in text
    assert "FILE_TYPE" not in docs[0]["text"]


def test_german_special_characters(tmp_path: Path) -> None:
    """Unicode and German characters survive the round trip unchanged."""
    name = "1.05.01.038: Gebühren- und Rechnungsabschlussreklamation bearbeiten"
    content = "Größe, Maßnahmen, Übergänge und Süße müssen geprüft werden."
    _, processes, docs, _ = _run(
        tmp_path,
        models=[_model("m_ä", name)],
        hierarchy=[],
        documents=[_document("d_ü", "Größenleitfaden", content)],
    )
    assert f"PROCESS_NAME: {name}" in processes[0]["text"]
    assert content in docs[0]["text"]


def test_empty_and_malformed_jsonl(tmp_path: Path) -> None:
    """Blank lines are skipped; malformed JSON fails explicitly."""
    models_path = tmp_path / "models.jsonl"
    models_path.write_text(
        '{"model_id": "m_x", "name": "Process X"}\n'
        "\n"
        '{"model_id": "", "name": ""}\n'
        '{"name": "Ohne ID"}\n',
        encoding="utf-8",
    )
    _write_jsonl(tmp_path / "hierarchy.jsonl", [])
    _write_jsonl(tmp_path / "documents.jsonl", [])
    config = ConverterConfig(
        models_path=models_path,
        hierarchy_path=tmp_path / "hierarchy.jsonl",
        documents_path=tmp_path / "documents.jsonl",
        output_dir=tmp_path / "output",
    )
    report = run_conversion(config)
    assert report["inputs"]["models"]["valid"] == 1
    assert report["inputs"]["models"]["rejected"] == 2
    assert report["inputs"]["models"]["blank_lines"] == 1

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text('{"model_id": "m_x"\n', encoding="utf-8")
    bad_config = ConverterConfig(
        models_path=bad_path,
        hierarchy_path=tmp_path / "hierarchy.jsonl",
        documents_path=tmp_path / "documents.jsonl",
        output_dir=tmp_path / "output2",
    )
    with pytest.raises(SchemaError):
        run_conversion(bad_config)


def test_duplicate_model_ids_fail_explicitly(tmp_path: Path) -> None:
    """Duplicate authoritative IDs threaten integrity and must fail."""
    with pytest.raises(ConversionError):
        _run(
            tmp_path,
            models=[_model("m_x", "A"), _model("m_x", "B")],
            hierarchy=[],
            documents=[],
        )


def test_unknown_relation_type_reported_not_textualized(tmp_path: Path) -> None:
    """Unexpected relation types are preserved in the audit, not mapped."""
    report, processes, _, ground_truth = _run(
        tmp_path,
        models=[_model("m_x", "Process X"), _model("m_y", "Process Y")],
        hierarchy=[_relation("m_x", "m_y", "C_SOMETHING_ELSE")],
        documents=[],
    )
    assert ground_truth == []
    assert len(report["relationships"]["unknown_types"]) == 1
    assert (
        report["relationships"]["unknown_types"][0]["relation_type"]
        == "C_SOMETHING_ELSE"
    )
    assert report["relationships"]["unknown_types"][0]["textualized"] is False
    assert "RELATIONSHIPS" not in processes[0]["text"]


def test_output_determinism(tmp_path: Path) -> None:
    """Two runs on identical inputs produce byte-identical outputs."""
    models = [
        _model("m_x", "Process X"),
        _model("m_y", "Process Y"),
        _model("m_z", "Process Z"),
    ]
    hierarchy = [_relation("m_x", "m_y"), _relation("m_y", "m_z", "C_CROSS_REFERENCE")]
    documents = [_document("d1", "Doc", "Inhalt eines Dokuments.")]
    _, _, _, _ = _run(tmp_path / "run1", models, hierarchy, documents)
    _, _, _, _ = _run(tmp_path / "run2", models, hierarchy, documents)
    out1 = tmp_path / "run1" / "output"
    out2 = tmp_path / "run2" / "output"
    for name in (
        "processes_textualized.jsonl",
        "documents_textualized.jsonl",
        "relationship_ground_truth.jsonl",
        "metadata_mapping.jsonl",
        "entity_lookup.json",
    ):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name


def test_combined_text_export(tmp_path: Path) -> None:
    """Combined files split back into exactly one array element per record."""
    report, processes, docs, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X"), _model("m_y", "Process Y")],
        hierarchy=[_relation("m_x", "m_y")],
        documents=[
            _document("d1", "Doc A", "Inhalt A."),
            _document("d2", "Doc B", "Inhalt B."),
        ],
        export_combined_text=True,
    )
    out = tmp_path / "output"
    proc_text = (out / "processes_text.txt").read_text(encoding="utf-8")
    doc_text = (out / "documents_text.txt").read_text(encoding="utf-8")
    proc_array = proc_text.split(f"\n{COMBINED_RECORD_DELIMITER}\n")
    doc_array = doc_text.split(f"\n{COMBINED_RECORD_DELIMITER}\n")
    assert proc_array == [rec["text"] for rec in processes]
    assert doc_array == [rec["text"] for rec in docs]
    # No JSON serialization syntax leaks into the combined text.
    assert '"record_id"' not in proc_text
    assert not proc_text.startswith("{")
    assert report["outputs"]["processes_combined"] == str(out / "processes_text.txt")
    assert report["outputs"]["documents_combined"] == str(out / "documents_text.txt")


def test_combined_text_absent_by_default(tmp_path: Path) -> None:
    """Combined text files are only written when explicitly requested."""
    report, _, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[_document("d1", "Doc", "Inhalt.")],
    )
    out = tmp_path / "output"
    assert not (out / "processes_text.txt").exists()
    assert not (out / "documents_text.txt").exists()
    assert report["outputs"]["processes_combined"] is None
    assert report["outputs"]["documents_combined"] is None


def test_exclude_file_type(tmp_path: Path) -> None:
    """Excluded file types are skipped entirely and reported, not dropped silently."""
    report, _, docs, _ = _run(
        tmp_path,
        models=[],
        hierarchy=[],
        documents=[
            _document("d1", "PDF-Doc", "PDF Inhalt.", file_type="pdf"),
            _document("d2", "Excel-Doc", "Excel Inhalt.", file_type="xlsx"),
            _document("d3", "Upper XLSX", "Noch ein Excel.", file_type="XLSX"),
            _document("d4", "Typelos", "Ohne Typ."),
        ],
        excluded_file_types=("xlsx",),
    )
    assert [rec["source_id"] for rec in docs] == ["d1", "d4"]
    assert report["documents"]["source_documents"] == 2
    excluded = report["documents"]["excluded_by_file_type"]
    assert [entry["document_id"] for entry in excluded] == ["d2", "d3"]
    assert report["inputs"]["documents"]["valid"] == 4
    assert any("Excluded 2 document(s)" in w for w in report["validation"]["warnings"])


def test_no_file_types_excluded_by_default(tmp_path: Path) -> None:
    """The default configuration excludes no file types, including xlsx."""
    report, _, docs, _ = _run(
        tmp_path,
        models=[],
        hierarchy=[],
        documents=[
            _document("d1", "PDF-Doc", "PDF Inhalt.", file_type="pdf"),
            _document("d2", "Excel-Doc", "Excel Inhalt.", file_type="xlsx"),
        ],
    )
    assert [rec["source_id"] for rec in docs] == ["d1", "d2"]
    assert report["documents"]["excluded_by_file_type"] == []


def test_real_sample_smoke(tmp_path: Path) -> None:
    """Smoke-test against the real 40% sample when it is available."""
    data_dir = Path(__file__).resolve().parents[4] / "test-data" / "test-data_40pct"
    if not (data_dir / "models_40pct.jsonl").is_file():
        pytest.skip("real 40% sample not available")
    config = ConverterConfig(
        models_path=data_dir / "models_40pct.jsonl",
        hierarchy_path=data_dir / "hierarchy_40pct.jsonl",
        documents_path=data_dir / "documents_40pct.jsonl",
        output_dir=tmp_path / "output",
        limit=25,
    )
    report = run_conversion(config)
    assert report["validation"]["errors"] == []
    assert report["processes"]["records"] == 25


def test_addon_description_enriches_process_record(tmp_path: Path) -> None:
    """Overview-chunk descriptions land in PROCESS_DESCRIPTION."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("{ABC-123}", "Process X")],
        hierarchy=[],
        documents=[],
        descriptions=[_overview_chunk("abc-123", "Der Prozess beschreibt X.")],
    )
    text = processes[0]["text"]
    assert "PROCESS_DESCRIPTION:" in text
    assert "Der Prozess beschreibt X." in text
    assert "Verantwortlich" not in text.split("PROCESS_DESCRIPTION:")[1]
    assert processes[0]["metadata"]["description_source"] == "description_addon"
    stats = report["inputs"]["descriptions"]
    assert stats["matched_models"] == 1
    assert stats["unmatched_model_ids"] == []
    assert report["processes"]["description_source_counts"] == {
        "models_export": 0,
        "description_addon": 1,
        "none": 0,
    }


def test_addon_join_ignores_non_overview_chunks(tmp_path: Path) -> None:
    """Lane/activity chunks never contribute descriptions."""
    lane = _overview_chunk("abc-123", "Sollte ignoriert werden.")
    lane["object_type"] = "lane"
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("abc-123", "Process X")],
        hierarchy=[],
        documents=[],
        descriptions=[lane],
    )
    assert "PROCESS_DESCRIPTION" not in processes[0]["text"]
    assert report["inputs"]["descriptions"]["overview_rows"] == 0
    assert report["processes"]["description_source_counts"]["none"] == 1


def test_addon_empty_description_not_emitted(tmp_path: Path) -> None:
    """'Kein Eintrag' and blank descriptions are treated as missing."""
    blank = _overview_chunk("m_b", None)
    blank["text"] = "Prozessmodell: Y\nBeschreibung / Zweck: \n \nQuelle: https://x\n"
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_a", "Process A"), _model("m_b", "Process B")],
        hierarchy=[],
        documents=[],
        descriptions=[_overview_chunk("m_a", None), blank],
    )
    assert all("PROCESS_DESCRIPTION" not in p["text"] for p in processes)
    assert report["inputs"]["descriptions"]["empty_descriptions"] == 2
    assert report["processes"]["description_source_counts"]["none"] == 2


def test_addon_multiline_description_stops_at_next_label(tmp_path: Path) -> None:
    """Multi-paragraph descriptions survive; following fields are cut off."""
    chunk = _overview_chunk("abc-123", "ignored")
    chunk["text"] = (
        "Prozessmodell: 1.0: X\n"
        "Version: 10.00\n"
        "Beschreibung / Zweck: Erster Absatz.\n"
        " \n"
        "Zweiter Absatz mit Details.\n"
        "Verantwortlich: Max\n"
        "Attribute:\n"
        "  Zweck: Erster Absatz.\n"
    )
    _, processes, _, _ = _run(
        tmp_path,
        models=[_model("abc-123", "Process X")],
        hierarchy=[],
        documents=[],
        descriptions=[chunk],
    )
    text = processes[0]["text"]
    assert "Erster Absatz.\nZweiter Absatz mit Details." in text
    assert "Verantwortlich" not in text
    assert "Attribute" not in text


def test_addon_unmatched_models_reported_not_invented(tmp_path: Path) -> None:
    """Descriptions for unknown models are reported, never textualized."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[],
        descriptions=[
            _overview_chunk("m_x", "Passt."),
            _overview_chunk("m_ghost", "Gehört nirgendwo hin."),
        ],
    )
    assert "Passt." in processes[0]["text"]
    assert "Gehört nirgendwo hin." not in processes[0]["text"]
    stats = report["inputs"]["descriptions"]
    assert stats["unmatched_model_ids"] == ["m_ghost"]
    assert any(
        "not in the models export" in w for w in report["validation"]["warnings"]
    )


def test_models_export_description_wins_over_addon(tmp_path: Path) -> None:
    """A real models-export description takes precedence over the addon."""
    model = _model(
        "m_x",
        "Process X",
        description="Echte Beschreibung aus dem Export.",
        json_string='{"metaName": "MT_BUSINESS_PROCESS_DIAGRAM_BPMN_20"}',
    )
    report, processes, _, _ = _run(
        tmp_path,
        models=[model],
        hierarchy=[],
        documents=[],
        descriptions=[_overview_chunk("m_x", "Addon-Beschreibung.")],
    )
    text = processes[0]["text"]
    assert "Echte Beschreibung aus dem Export." in text
    assert "Addon-Beschreibung." not in text
    assert processes[0]["metadata"]["description_source"] == "models_export"
    assert report["processes"]["description_source_counts"]["models_export"] == 1


def test_addon_jsonl_format_accepted(tmp_path: Path) -> None:
    """The addon reader also accepts newline-delimited JSON."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[],
    )
    assert report["inputs"]["descriptions"] is None
    # Same run, but with a JSONL addon file.
    addon_path = tmp_path / "descriptions.jsonl"
    _write_jsonl(addon_path, [_overview_chunk("m_x", "JSONL-Beschreibung.")])
    report2, processes2, _, _ = _run(
        tmp_path / "second",
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[],
        descriptions_path=addon_path,
    )
    assert "JSONL-Beschreibung." in processes2[0]["text"]
    assert report2["inputs"]["descriptions"]["matched_models"] == 1


def test_real_sample_addon_smoke(tmp_path: Path) -> None:
    """The real addon enriches exactly the expected real-sample models."""
    from adonis_converter.description_addon import (
        extract_overview_description,
        normalize_model_key,
        read_description_chunks,
    )
    from adonis_converter.schema import read_jsonl

    data_dir = Path(__file__).resolve().parents[4] / "test-data" / "test-data_40pct"
    models_path = data_dir / "models_40pct.jsonl"
    addon_path = data_dir / "description_hierarchy-addon.json"
    if not models_path.is_file() or not addon_path.is_file():
        pytest.skip("real 40% sample or descriptions addon not available")
    config = ConverterConfig(
        models_path=models_path,
        hierarchy_path=data_dir / "hierarchy_40pct.jsonl",
        documents_path=data_dir / "documents_40pct.jsonl",
        descriptions_path=addon_path,
        output_dir=tmp_path / "output",
        limit=25,
    )
    report = run_conversion(config)
    assert report["validation"]["errors"] == []

    # Independent expectation computed straight from the real files.
    model_keys = {
        normalize_model_key(row.data["model_id"])
        for row in read_jsonl(models_path, limit=25).rows
    }
    descriptions_by_key: dict[str, str] = {}
    for row in read_description_chunks(addon_path):
        if row.get("object_type") != "overview":
            continue
        description = extract_overview_description(row.get("text"))
        if description:
            descriptions_by_key.setdefault(
                normalize_model_key(row.get("model_id")), description
            )
    expected = len(model_keys & descriptions_by_key.keys())
    stats = report["inputs"]["descriptions"]
    assert stats["rows_read"] == 7819
    assert stats["overview_rows"] == 567
    # With limit=25 only 25 models are loaded, so the remaining addon
    # descriptions are correctly reported as unmatched, never invented.
    assert stats["unmatched_model_ids"] == sorted(
        descriptions_by_key.keys() - model_keys
    )
    assert stats["matched_models"] == expected
    assert report["processes"]["description_source_counts"]["description_addon"] == (
        expected
    )


def _act_sentence(process_name: str, activity_name: str, step: int | None) -> str:
    """Render the expected activity sentence for assertions."""
    if step is None:
        return f'Process "{process_name}" has activity "{activity_name}".'
    return f'Process "{process_name}" has activity "{activity_name}" (step {step}).'


def test_activities_appended_after_relationships(tmp_path: Path) -> None:
    """Activities form an ACTIVITIES section after CROSS_REFERENCES, step-ordered."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X"), _model("m_y", "Process Y")],
        hierarchy=[
            _relation("m_x", "m_y"),
            _relation("m_x", "m_y", relation_type="C_CROSS_REFERENCE"),
        ],
        documents=[],
        activities=[
            _activity_chunk("m_x", "Zweiter Schritt", 2),
            _activity_chunk("m_x", "Erster Schritt", 1),
        ],
    )
    text_x = next(p for p in processes if p["source_id"] == "m_x")["text"]
    assert "ACTIVITIES:" in text_x
    assert text_x.index("ACTIVITIES:") > text_x.index("CROSS_REFERENCES:")
    first = _act_sentence("Process X", "Erster Schritt", 1)
    second = _act_sentence("Process X", "Zweiter Schritt", 2)
    assert first in text_x
    assert second in text_x
    # Out-of-order input is sorted by step number.
    assert text_x.index(first) < text_x.index(second)
    meta_x = next(p for p in processes if p["source_id"] == "m_x")["metadata"]
    assert meta_x["activity_count"] == 2
    stats = report["inputs"]["activities"]
    assert stats["activity_rows"] == 2
    assert stats["matched_models"] == 1
    assert stats["matched_activities"] == 2
    assert report["processes"]["with_activities"] == 1


def test_activities_absent_by_default(tmp_path: Path) -> None:
    """Without an activities path no ACTIVITIES section is rendered."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[],
    )
    assert "ACTIVITIES" not in processes[0]["text"]
    assert "activity_count" not in processes[0]["metadata"]
    assert report["inputs"]["activities"] is None
    assert report["config"]["activities_path"] is None


def test_process_without_activities_has_no_section(tmp_path: Path) -> None:
    """A model with no activity chunks gets no ACTIVITIES section."""
    _, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X"), _model("m_y", "Process Y")],
        hierarchy=[],
        documents=[],
        activities=[_activity_chunk("m_x", "Einziger Schritt", 1)],
    )
    text_y = next(p for p in processes if p["source_id"] == "m_y")["text"]
    meta_y = next(p for p in processes if p["source_id"] == "m_y")["metadata"]
    assert "ACTIVITIES" not in text_y
    assert meta_y["activity_count"] == 0


def test_activity_join_normalizes_chunk_model_ids(tmp_path: Path) -> None:
    """model_{uuid}, the mangled model_{uuid artifact and nested metadata join."""
    _, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[],
        activities=[
            _activity_chunk("model_{m_x}", "Top Level", 1),
            _activity_chunk("model_{m_x", "Mangled Brace", 2),
            _activity_chunk("m_x", "Nested Metadata", 3, nested=True),
        ],
    )
    text_x = processes[0]["text"]
    for name, step in (("Top Level", 1), ("Mangled Brace", 2), ("Nested Metadata", 3)):
        assert _act_sentence("Process X", name, step) in text_x
    assert processes[0]["metadata"]["activity_count"] == 3


def test_activity_unmatched_models_reported_not_invented(tmp_path: Path) -> None:
    """Activities for unknown models are reported, never textualized."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[],
        activities=[
            _activity_chunk("m_x", "Bekannter Schritt", 1),
            _activity_chunk("m_unknown", "Fremder Schritt", 1),
        ],
    )
    assert "Fremder Schritt" not in processes[0]["text"]
    stats = report["inputs"]["activities"]
    assert stats["unmatched_model_ids"] == ["m_unknown"]
    assert stats["matched_activities"] == 1
    assert any(
        "not in the models export" in w for w in report["validation"]["warnings"]
    )


def test_activity_without_name_or_step_handled(tmp_path: Path) -> None:
    """Nameless chunks are skipped; missing steps render without '(step n)'."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[],
        activities=[
            _activity_chunk("m_x", None, 1),
            _activity_chunk("m_x", "Schritt Ohne Nummer", None),
        ],
    )
    text_x = processes[0]["text"]
    assert "Notiz" not in text_x
    assert _act_sentence("Process X", "Schritt Ohne Nummer", None) in text_x
    stats = report["inputs"]["activities"]
    assert stats["unnamed_activities"] == 1
    assert stats["matched_activities"] == 1


def test_activities_missing_file_fails(tmp_path: Path) -> None:
    """A configured but missing chunks file is an explicit SchemaError."""
    with pytest.raises(SchemaError):
        _run(
            tmp_path,
            models=[_model("m_x", "Process X")],
            hierarchy=[],
            documents=[],
            activities_path=tmp_path / "missing_chunks.jsonl",
        )


# ---------------------------------------------------------------------------
# Extra chunk types ("everything" mode: lane / compliance / other)
# ---------------------------------------------------------------------------

EVERYTHING_TYPES: tuple[str, ...] = ("lane", "compliance", "other")


def _lane_chunk(
    model_id: str,
    lane_name: str | None,
    number: int | None,
    kind: str = "C_LANE",
    nested: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal synthetic lane chunk row (chunks.jsonl shape).

    Args:
        model_id: Model reference (bare key; nested form adds the
            ``model_{uuid}`` metadata wrapper).
        lane_name: Lane name; ``None`` omits the metadata value.
        number: Lane number for the ``_lane_<n>`` node ID suffix;
            ``None`` omits the suffix.
        kind: ``object_meta_name`` / ``Typ:`` value (C_LANE or C_POOL).
        nested: Use the nested ``metadata_`` production shape.
        extra: Extra row fields.
    """
    node_suffix = f"_lane_{number}" if number is not None else ""
    text = (
        "Swimlane / Pool im Prozessmodell: Beispiel\n"
        f"Lane-Name: {lane_name or ''}\n"
        f"Typ: {kind}\n"
        "Modell-Status: Gültig\n"
        "Quelle: https://example.invalid/view\n"
    )
    if nested:
        row: dict[str, Any] = {
            "id": 30,
            "node_id": f"model_{{{model_id}}}{node_suffix}",
            "text": text,
            "metadata_": {
                "model_id": f"model_{{{model_id}}}",
                "object_type": "lane",
                "lane_name": lane_name or "",
                "object_meta_name": kind,
            },
        }
    else:
        row = {
            "id": 30,
            "node_id": f"model_{{{model_id}}}{node_suffix}",
            "model_id": model_id,
            "object_type": "lane",
            "lane_name": lane_name or "",
            "object_meta_name": kind,
            "text": text,
        }
    row.update(extra)
    return row


def _compliance_chunk(
    model_id: str,
    status: str = "",
    risk_level: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal synthetic compliance chunk row.

    Args:
        model_id: Model reference (bare key).
        status: ``status`` metadata value (``""`` = missing).
        risk_level: ``risk_level`` metadata value (``""`` = missing).
        extra: Extra row fields.
    """
    text = (
        "Compliance-Informationen für: Beispiel\n"
        f"Status: {status or 'Kein Eintrag'}\n"
        f"  Risikostufe: {risk_level or 'Kein Eintrag'}\n"
        "Quelle: https://example.invalid/view\n"
    )
    row = {
        "id": 31,
        "model_id": model_id,
        "object_type": "compliance",
        "status": status,
        "risk_level": risk_level,
        "text": text,
    }
    row.update(extra)
    return row


def _other_chunk(model_id: str, entries: list[str], **extra: Any) -> dict[str, Any]:
    """Build a minimal synthetic related-objects ("other") chunk row.

    Args:
        model_id: Model reference (bare key).
        entries: Listing entries without the leading bullet, e.g.
            ``"1.0: Sub [C_SUB_PROCESS]"``.
        extra: Extra row fields.
    """
    lines = ["Weitere Objekte in: Beispiel"]
    lines.extend(f"  - {entry}" for entry in entries)
    lines.append("Quelle: https://example.invalid/view")
    row = {
        "id": 32,
        "model_id": model_id,
        "object_type": "other",
        "text": "\n".join(lines) + "\n",
    }
    row.update(extra)
    return row


def test_everything_renders_all_sections_after_activities(tmp_path: Path) -> None:
    """LANES/COMPLIANCE/RELATED_OBJECTS follow ACTIVITIES, sentences rendered."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X"), _model("m_y", "Process Y")],
        hierarchy=[_relation("m_x", "m_y")],
        documents=[],
        activities=[
            _activity_chunk("m_x", "Einziger Schritt", 1),
            _lane_chunk("m_x", "Fachbereich", 0),
            _lane_chunk("m_x", "Zentrale", 1, kind="C_POOL"),
            _compliance_chunk("m_x", status="Gültig", risk_level="Hoch"),
            _other_chunk(
                "m_x",
                [
                    "1.0: Sub [C_SUB_PROCESS]",
                    "2.0: Ref [C_CROSS_REFERENCE]",
                    "Änderungsbedarf liegt vor [C_START_EVENT]",
                    "Ungetaggte Zeile",
                ],
            ),
        ],
        extra_chunk_types=EVERYTHING_TYPES,
    )
    text_x = next(p for p in processes if p["source_id"] == "m_x")["text"]
    # Section order: ACTIVITIES < LANES < COMPLIANCE < RELATED_OBJECTS.
    assert text_x.index("ACTIVITIES:") < text_x.index("LANES:")
    assert text_x.index("LANES:") < text_x.index("COMPLIANCE:")
    assert text_x.index("COMPLIANCE:") < text_x.index("RELATED_OBJECTS:")
    # Sentence rendering per type.
    assert 'Process "Process X" has swimlane "Fachbereich" (Nr. 0).' in text_x
    assert 'Process "Process X" has pool "Zentrale" (Nr. 1).' in text_x
    assert 'Process "Process X" has the status "Gültig".' in text_x
    assert 'Process "Process X" has the risk level "Hoch".' in text_x
    assert 'Process "Process X" includes "1.0: Sub" as subprocess.' in text_x
    assert 'Process "Process X" includes "2.0: Ref" as cross-reference.' in text_x
    assert (
        'Process "Process X" has "Änderungsbedarf liegt vor" as START_EVENT.' in text_x
    )
    assert 'Process "Process X" has "Ungetaggte Zeile" as related object.' in text_x
    # Headers and source URLs are noise and never textualized.
    assert "Quelle" not in text_x
    assert "Weitere Objekte in" not in text_x
    assert "Compliance-Informationen" not in text_x
    assert "Swimlane / Pool im Prozessmodell" not in text_x
    # Metadata counts and report stats.
    meta_x = next(p for p in processes if p["source_id"] == "m_x")["metadata"]
    assert meta_x["lane_count"] == 2
    assert meta_x["has_compliance"] is True
    assert meta_x["related_object_count"] == 4
    stats = report["inputs"]["activities"]["extras"]
    assert stats["extra_chunk_types"] == list(EVERYTHING_TYPES)
    assert stats["rows_by_type"] == {"lane": 2, "compliance": 1, "other": 1}
    assert stats["matched_lanes"] == 2
    assert stats["matched_compliance"] == 1
    assert stats["matched_related_objects"] == 4
    assert report["processes"]["with_lanes"] == 1
    assert report["processes"]["with_compliance"] == 1
    assert report["processes"]["with_related_objects"] == 1
    assert report["config"]["extra_chunk_types"] == list(EVERYTHING_TYPES)
    # The other model gets no extra sections.
    text_y = next(p for p in processes if p["source_id"] == "m_y")["text"]
    assert "LANES" not in text_y
    assert "COMPLIANCE" not in text_y
    assert "RELATED_OBJECTS" not in text_y


def test_extra_chunk_types_absent_by_default(tmp_path: Path) -> None:
    """Without extra_chunk_types, lane/compliance/other chunks are ignored."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[],
        activities=[
            _activity_chunk("m_x", "Einziger Schritt", 1),
            _lane_chunk("m_x", "Fachbereich", 0),
            _compliance_chunk("m_x", status="Gültig"),
            _other_chunk("m_x", ["Start [C_START_EVENT]"]),
        ],
    )
    text_x = processes[0]["text"]
    assert "ACTIVITIES:" in text_x
    assert "LANES" not in text_x
    assert "COMPLIANCE" not in text_x
    assert "RELATED_OBJECTS" not in text_x
    assert "lane_count" not in processes[0]["metadata"]
    assert report["inputs"]["activities"].get("extras") is None
    assert report["config"]["extra_chunk_types"] == []


def test_extra_chunk_types_require_activities_path(tmp_path: Path) -> None:
    """extra_chunk_types without activities_path is a config ValueError."""
    with pytest.raises(ValueError, match="requires activities_path"):
        ConverterConfig(
            models_path=tmp_path / "models.jsonl",
            hierarchy_path=tmp_path / "hierarchy.jsonl",
            documents_path=tmp_path / "documents.jsonl",
            output_dir=tmp_path / "output",
            extra_chunk_types=("lane",),
        )


def test_unknown_extra_chunk_type_rejected(tmp_path: Path) -> None:
    """Unknown extra chunk types are a config ValueError."""
    with pytest.raises(ValueError, match="Unsupported extra chunk type"):
        ConverterConfig(
            models_path=tmp_path / "models.jsonl",
            hierarchy_path=tmp_path / "hierarchy.jsonl",
            documents_path=tmp_path / "documents.jsonl",
            output_dir=tmp_path / "output",
            activities_path=tmp_path / "chunks.jsonl",
            extra_chunk_types=("lane", "bogus"),
        )


def test_compliance_placeholder_values_skipped(tmp_path: Path) -> None:
    """Empty/Kein Eintrag compliance values render no sentence."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X"), _model("m_y", "Process Y")],
        hierarchy=[],
        documents=[],
        activities=[
            # m_x: only a real status; the risk level is the placeholder.
            _compliance_chunk("m_x", status="Gültig", risk_level="Kein Eintrag"),
            # m_y: both values are placeholders -> no COMPLIANCE section.
            _compliance_chunk("m_y", status="", risk_level="Kein Eintrag"),
        ],
        extra_chunk_types=EVERYTHING_TYPES,
    )
    text_x = next(p for p in processes if p["source_id"] == "m_x")["text"]
    assert 'Process "Process X" has the status "Gültig".' in text_x
    assert "risk level" not in text_x
    text_y = next(p for p in processes if p["source_id"] == "m_y")["text"]
    assert "COMPLIANCE" not in text_y
    stats = report["inputs"]["activities"]["extras"]
    assert stats["empty_rows"] == 1
    assert stats["matched_compliance"] == 1


def test_extra_chunks_unmatched_models_reported(tmp_path: Path) -> None:
    """Extra chunks for unknown models are reported, never textualized."""
    report, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[],
        activities=[
            _lane_chunk("m_x", "Fachbereich", 0),
            _lane_chunk("m_unknown", "Fremde Bahn", 0),
            _compliance_chunk("m_unknown", status="Gültig"),
        ],
        extra_chunk_types=EVERYTHING_TYPES,
    )
    text_x = processes[0]["text"]
    assert "Fremde Bahn" not in text_x
    stats = report["inputs"]["activities"]["extras"]
    assert stats["unmatched_model_ids"] == ["m_unknown"]
    assert stats["matched_lanes"] == 1
    assert any(
        "not in the models export" in w for w in report["validation"]["warnings"]
    )


def test_lane_without_number_or_metadata_falls_back_to_text(tmp_path: Path) -> None:
    """Lanes without a node ID number render without the Nr. suffix; nested rows join."""
    _, processes, _, _ = _run(
        tmp_path,
        models=[_model("m_x", "Process X")],
        hierarchy=[],
        documents=[],
        activities=[
            # No _lane_<n> suffix -> no (Nr. ...) suffix.
            _lane_chunk("m_x", "Namenslose Nummer", None),
            # Nested production shape with metadata_ join.
            _lane_chunk("m_x", "Geschachtelt", 2, kind="C_POOL", nested=True),
        ],
        extra_chunk_types=EVERYTHING_TYPES,
    )
    text_x = processes[0]["text"]
    assert 'Process "Process X" has swimlane "Namenslose Nummer".' in text_x
    assert "(Nr." not in text_x.split("Namenslose Nummer")[0].split("LANES:")[-1]
    assert 'Process "Process X" has pool "Geschachtelt" (Nr. 2).' in text_x
    assert processes[0]["metadata"]["lane_count"] == 2
