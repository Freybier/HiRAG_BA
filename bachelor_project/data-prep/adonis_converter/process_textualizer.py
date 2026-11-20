"""Parent-only textualization of ADONIS process models.

Every process produces exactly one textual record. Verified outgoing
relationships are embedded in the *source* process's record only. Child
records never repeat that they are a subprocess of their parent, no
inverse statements are generated, and grandchildren are never listed as
direct subprocesses.

The record text carries only the per-process semantics: the model's name,
its description/purpose body, and its subprocesses/cross-references.
``metaName``/``type`` are deliberately not textualized — they collapse to
one or two constant values across the whole corpus and do not
differentiate processes (they remain available in the sidecar metadata).
Process IDs stay in the JSONL metadata (``source_id``), not in the text.
With ``ConverterConfig.activities_path`` set, the model's activity steps
(from the activity chunks of the chunks corpus) are appended as an
``ACTIVITIES`` section after the relationship sections. With
``ConverterConfig.extra_chunk_types`` set (the "everything" mode), the
model's lanes, compliance values and related-objects listing are
additionally appended as ``LANES``, ``COMPLIANCE`` and
``RELATED_OBJECTS`` sections after ``ACTIVITIES``.
"""

from __future__ import annotations

from .activity_addon import Activity, ExtraChunks, Lane, RelatedObject
from .config import ConverterConfig
from .relationship_mapper import CATEGORY_CROSS_REFERENCE, CanonicalRelationship
from .schema import ProcessModel


def subprocess_sentence(parent_name: str, child_name: str) -> str:
    """Render the canonical subprocess sentence.

    Args:
        parent_name: Authoritative name of the parent process.
        child_name: Authoritative name of the subprocess.

    Returns:
        The fully written-out relationship sentence (names only; IDs live
        in the JSONL metadata, not in the text).
    """
    return f'Process "{parent_name}" has subprocess "{child_name}".'


def cross_reference_sentence(source_name: str, target_name: str) -> str:
    """Render the canonical cross-reference sentence.

    Args:
        source_name: Authoritative name of the source process.
        target_name: Authoritative name of the referenced process.

    Returns:
        The fully written-out relationship sentence (names only; IDs live
        in the JSONL metadata, not in the text).
    """
    return f'Process "{source_name}" cross-references "{target_name}".'


def relationship_sentence(
    relationship: CanonicalRelationship,
    source: ProcessModel,
    target: ProcessModel,
) -> str:
    """Render one canonical relationship as a sentence.

    Args:
        relationship: The canonical relationship to render.
        source: The authoritative source process model.
        target: The authoritative target process model.

    Returns:
        The relationship sentence for the relationship's category.
    """
    if relationship.category == CATEGORY_CROSS_REFERENCE:
        return cross_reference_sentence(source.name, target.name)
    return subprocess_sentence(source.name, target.name)


def activity_sentence(process_name: str, activity: Activity) -> str:
    """Render one process activity step as a sentence.

    Args:
        process_name: Authoritative name of the owning process.
        activity: The activity step extracted from the chunks corpus.

    Returns:
        The activity sentence, including the step number when the chunk
        metadata carried one (names only; IDs live in the JSONL metadata,
        not in the text).
    """
    if activity.step_number is None:
        return f'Process "{process_name}" has activity "{activity.name}".'
    return (
        f'Process "{process_name}" has activity "{activity.name}" '
        f"(step {activity.step_number})."
    )


def lane_sentence(process_name: str, lane: Lane) -> str:
    """Render one swimlane/pool of a process as a sentence.

    Args:
        process_name: Authoritative name of the owning process.
        lane: The lane extracted from the chunks corpus.

    Returns:
        The lane sentence, including the lane number (from the chunk's
        ``_lane_<n>`` node ID suffix) when available.
    """
    number_suffix = f" (Nr. {lane.number})" if lane.number is not None else ""
    return f'Process "{process_name}" has {lane.kind} "{lane.name}"{number_suffix}.'


def compliance_status_sentence(process_name: str, status: str) -> str:
    """Render the process's compliance status as a sentence.

    Args:
        process_name: Authoritative name of the owning process.
        status: Model status from the compliance chunk.

    Returns:
        The compliance status sentence.
    """
    return f'Process "{process_name}" has the status "{status}".'


def compliance_risk_level_sentence(process_name: str, risk_level: str) -> str:
    """Render the process's risk level as a sentence.

    Args:
        process_name: Authoritative name of the owning process.
        risk_level: Risk level from the compliance chunk.

    Returns:
        The risk level sentence.
    """
    return f'Process "{process_name}" has the risk level "{risk_level}".'


def related_object_sentence(process_name: str, related_object: RelatedObject) -> str:
    """Render one related-objects listing entry as a sentence.

    Subprocess and cross-reference tags keep the ``includes`` wording of
    the relationship sections; every other tag is rendered with ``has``
    and the tag label without its ``C_`` prefix (e.g. ``START_EVENT``).

    Args:
        process_name: Authoritative name of the owning process.
        related_object: One entry of the related-objects listing.

    Returns:
        The related-object sentence.
    """
    if related_object.tag == "C_SUB_PROCESS":
        return (
            f'Process "{process_name}" includes "{related_object.name}" as subprocess.'
        )
    if related_object.tag == "C_CROSS_REFERENCE":
        return (
            f'Process "{process_name}" includes "{related_object.name}" '
            "as cross-reference."
        )
    label = (
        related_object.tag.removeprefix("C_")
        if related_object.tag
        else "related object"
    )
    return f'Process "{process_name}" has "{related_object.name}" as {label}.'


def effective_description(
    model: ProcessModel, addon_description: str | None = None
) -> str | None:
    """Resolve the description to textualize for a process.

    The models export's ``description`` column only holds the ADONIS
    metaName constant, not free text, so a real export description wins
    whenever it differs from the metaName; otherwise the overview-chunk
    description from the ``data_vector_store2`` addon is used.

    Args:
        model: The process model to describe.
        addon_description: Description extracted from the addon's
            overview chunk for this model, if available.

    Returns:
        The description text, or ``None`` when neither source has one.
    """
    if model.description and model.description != model.meta_name:
        return model.description
    return addon_description


def description_source(
    model: ProcessModel, addon_description: str | None = None
) -> str:
    """Report which source supplied the textualized description.

    Args:
        model: The process model being rendered.
        addon_description: Description extracted from the addon's
            overview chunk for this model, if available.

    Returns:
        ``"models_export"``, ``"description_addon"`` or ``"none"``.
    """
    if model.description and model.description != model.meta_name:
        return "models_export"
    if addon_description:
        return "description_addon"
    return "none"


def render_process_text(
    model: ProcessModel,
    outgoing: list[CanonicalRelationship],
    models_by_id: dict[str, ProcessModel],
    config: ConverterConfig,
    addon_description: str | None = None,
    activities: list[Activity] | None = None,
    extras: ExtraChunks | None = None,
) -> str:
    """Render the parent-only textual record for one process.

    The record contains only the semantic sections ``PROCESS_NAME``,
    ``PROCESS_DESCRIPTION``, ``SUBPROCESSES``, ``CROSS_REFERENCES``,
    ``ACTIVITIES``, ``LANES``, ``COMPLIANCE`` and ``RELATED_OBJECTS``;
    all but the name are omitted when empty. In particular, no "has no
    subprocesses" statements are generated: the export is a sample, so
    missing rows do not prove missing relationships.
    ``metaName``/``type``/hierarchy depth are intentionally excluded
    (constant or non-semantic; still available in the sidecar metadata).

    Args:
        model: The process model to textualize.
        outgoing: The model's canonical outgoing relationships.
        models_by_id: All models indexed by authoritative ID.
        config: Converter configuration.
        addon_description: Description from the ``data_vector_store2``
            addon's overview chunk, used when the models export carries
            no real description.
        activities: The model's activity steps from the chunks corpus
            (only supplied when ``config.activities_path`` is set),
            rendered after the relationship sections.
        extras: The model's lane/compliance/related-objects chunk
            content (only supplied when ``config.extra_chunk_types`` is
            set), rendered after the ``ACTIVITIES`` section.

    Returns:
        The record text (plain text, no JSON serialization syntax).
    """
    lines: list[str] = [f"PROCESS_NAME: {model.name}"]

    # The real export stores the ADONIS metaName in the description column,
    # not free text. Only emit a description section when a source carries
    # actual descriptive content (see effective_description).
    description = effective_description(model, addon_description)
    if description:
        lines.append("")
        lines.append("PROCESS_DESCRIPTION:")
        lines.append(description)

    subprocesses = [rel for rel in outgoing if rel.category != CATEGORY_CROSS_REFERENCE]
    cross_references = [
        rel for rel in outgoing if rel.category == CATEGORY_CROSS_REFERENCE
    ]
    if subprocesses:
        lines.append("")
        lines.append("SUBPROCESSES:")
        for rel in subprocesses:
            lines.append(relationship_sentence(rel, model, models_by_id[rel.target_id]))
    if cross_references:
        lines.append("")
        lines.append("CROSS_REFERENCES:")
        for rel in cross_references:
            lines.append(relationship_sentence(rel, model, models_by_id[rel.target_id]))
    if activities:
        lines.append("")
        lines.append("ACTIVITIES:")
        for activity in activities:
            lines.append(activity_sentence(model.name, activity))
    if extras:
        if extras.lanes:
            lines.append("")
            lines.append("LANES:")
            for lane in extras.lanes:
                lines.append(lane_sentence(model.name, lane))
        if extras.compliance_status or extras.compliance_risk_level:
            lines.append("")
            lines.append("COMPLIANCE:")
            if extras.compliance_status:
                lines.append(
                    compliance_status_sentence(model.name, extras.compliance_status)
                )
            if extras.compliance_risk_level:
                lines.append(
                    compliance_risk_level_sentence(
                        model.name, extras.compliance_risk_level
                    )
                )
        if extras.related_objects:
            lines.append("")
            lines.append("RELATED_OBJECTS:")
            for related_object in extras.related_objects:
                lines.append(related_object_sentence(model.name, related_object))
    return "\n".join(lines) + "\n"


def sanitize_filename(record_id: str) -> str:
    """Make a record ID safe for use as a plain-text file name.

    Args:
        record_id: The record identifier, which may contain characters such
            as the braces of braced ADONIS UUIDs.

    Returns:
        A filesystem-safe file stem for Windows and Linux.
    """
    unsafe = '<>:"/\\|?*{}'
    cleaned = "".join("_" if char in unsafe else char for char in record_id)
    return cleaned.strip() or "record"
