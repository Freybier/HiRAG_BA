"""Canonical mapping of verified ADONIS hierarchy relationships.

The mapper turns raw hierarchy rows into deduplicated, canonical
relationships keyed by ``(source_id, relation_type, target_id)``. It never
invents relationships, never reverses directions and never infers links
from names or descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ConverterConfig
from .schema import ProcessModel, RelationshipRow

#: Relation categories understood by the textualizers.
CATEGORY_SUBPROCESS: str = "subprocess"
CATEGORY_CROSS_REFERENCE: str = "cross_reference"


@dataclass(frozen=True)
class CanonicalRelationship:
    """One verified, deduplicated relationship."""

    source_id: str
    target_id: str
    relation_type: str
    category: str
    occurrences: int
    first_line: int
    source_file: str

    # Return the deduplication key for this relationship.
    @property
    def canonical_key(self) -> tuple[str, str, str]:
        """Return the deduplication key for this relationship."""
        return (self.source_id, self.relation_type, self.target_id)


@dataclass
class RelationshipIndex:
    """Indexed view of all canonical relationships."""

    outgoing: dict[str, list[CanonicalRelationship]] = field(default_factory=dict)
    ground_truth: list[CanonicalRelationship] = field(default_factory=list)
    duplicates: list[dict[str, object]] = field(default_factory=list)
    unknown_types: list[dict[str, object]] = field(default_factory=list)
    broken_references: list[dict[str, object]] = field(default_factory=list)
    missing_endpoint_ids: list[dict[str, object]] = field(default_factory=list)
    name_mismatches: list[dict[str, object]] = field(default_factory=list)

    # Return the canonical outgoing relationships of a process.
    def outgoing_for(self, source_id: str) -> list[CanonicalRelationship]:
        """Return the canonical outgoing relationships of a process."""
        return self.outgoing.get(source_id, [])


# Map a relation type to its textualization category.
def categorize(relation_type: str, config: ConverterConfig) -> str | None:
    """Map a relation type to its textualization category.

    Args:
        relation_type: Original ADONIS relation type.
        config: Converter configuration with the explicit type mapping.

    Returns:
        The category, or ``None`` for unmapped (never guessed) types.
    """
    if relation_type in config.subprocess_relation_types:
        return CATEGORY_SUBPROCESS
    if relation_type in config.cross_reference_relation_types:
        return CATEGORY_CROSS_REFERENCE
    return None


# Cross-check the relationship direction against model-tree parents.
def check_direction(
    relationships: list[RelationshipRow],
    models_by_id: dict[str, ProcessModel],
    config: ConverterConfig,
) -> dict[str, object]:
    """Cross-check the relationship direction against model-tree parents.

    The models export carries a ``parent_model_id`` tree-containment field.
    If the hierarchy columns were reversed, we would expect the *parent*
    endpoint's tree parent to equal the child more often than the other way
    around. This check reports both counts; it never silently flips the
    direction.

    Args:
        relationships: Parsed hierarchy rows.
        models_by_id: Models indexed by their authoritative ID.
        config: Converter configuration.

    Returns:
        A report dictionary with agreement counts and a verdict.
    """
    forward = 0
    reversed_count = 0
    checked = 0
    for row in relationships:
        if row.relation_type not in config.subprocess_relation_types:
            continue
        if not row.source_id or not row.target_id:
            continue
        source = models_by_id.get(row.source_id)
        target = models_by_id.get(row.target_id)
        if source is None or target is None:
            continue
        checked += 1
        if target.tree_parent_id == row.source_id:
            forward += 1
        if source.tree_parent_id == row.target_id:
            reversed_count += 1
    verdict = "confirmed"
    if checked == 0:
        verdict = "ambiguous: no subprocess relationships to cross-check"
    elif reversed_count > forward:
        verdict = (
            "ambiguous: tree-parent agreement is stronger for the reversed "
            "direction; the hierarchy column mapping needs manual review"
        )
    return {
        "checked_subprocess_rows": checked,
        "forward_agreements": forward,
        "reversed_agreements": reversed_count,
        "verdict": verdict,
        "note": (
            "tree-parent agreement is only a partial signal because the "
            "ADONIS model-tree containment and subprocess relations are "
            "different relationship kinds; the column names "
            "parent_model_id/child_model_id remain the primary evidence"
        ),
    }


# Resolve relationship endpoints, deduplicate canonical edges, and audit missing or inconsistent
# data.
def build_relationship_index(
    relationships: list[RelationshipRow],
    models_by_id: dict[str, ProcessModel],
    config: ConverterConfig,
) -> RelationshipIndex:
    """Build the canonical relationship index from raw hierarchy rows.

    Deduplication uses the canonical key
    ``(source_id, relation_type, target_id)``. Duplicate occurrences are
    preserved in the audit data but textualized only once. Rows with
    missing endpoint IDs, broken references or unmapped relation types are
    reported, not textualized.

    Args:
        relationships: Parsed hierarchy rows in file order.
        models_by_id: Models indexed by their authoritative ID.
        config: Converter configuration.

    Returns:
        The canonical relationship index with full audit data.
    """
    index = RelationshipIndex()
    seen: dict[tuple[str, str, str], CanonicalRelationship] = {}
    for row in relationships:
        if not row.source_id or not row.target_id or not row.relation_type:
            index.missing_endpoint_ids.append(
                {
                    "source_file": row.source_file,
                    "line_number": row.line_number,
                    "source_id": row.source_id,
                    "target_id": row.target_id,
                    "relation_type": row.relation_type,
                }
            )
            continue
        category = categorize(row.relation_type, config)
        if category is None:
            index.unknown_types.append(
                {
                    "source_file": row.source_file,
                    "line_number": row.line_number,
                    "source_id": row.source_id,
                    "target_id": row.target_id,
                    "relation_type": row.relation_type,
                    "textualized": False,
                }
            )
            continue
        if row.source_id not in models_by_id or row.target_id not in models_by_id:
            index.broken_references.append(
                {
                    "source_file": row.source_file,
                    "line_number": row.line_number,
                    "source_id": row.source_id,
                    "target_id": row.target_id,
                    "relation_type": row.relation_type,
                    "missing_endpoint": (
                        "source" if row.source_id not in models_by_id else "target"
                    ),
                }
            )
            continue
        key = (row.source_id, row.relation_type, row.target_id)
        existing = seen.get(key)
        if existing is not None:
            seen[key] = CanonicalRelationship(
                source_id=existing.source_id,
                target_id=existing.target_id,
                relation_type=existing.relation_type,
                category=existing.category,
                occurrences=existing.occurrences + 1,
                first_line=existing.first_line,
                source_file=existing.source_file,
            )
            index.duplicates.append(
                {
                    "source_file": row.source_file,
                    "line_number": row.line_number,
                    "canonical_key": list(key),
                    "first_seen_line": existing.first_line,
                }
            )
            continue
        seen[key] = CanonicalRelationship(
            source_id=row.source_id,
            target_id=row.target_id,
            relation_type=row.relation_type,
            category=category,
            occurrences=1,
            first_line=row.line_number,
            source_file=row.source_file,
        )
        # Name consistency: the models table is authoritative; the exported
        # hierarchy names are only compared, never used for textualization.
        source = models_by_id[row.source_id]
        target = models_by_id[row.target_id]
        if (
            row.source_name_exported is not None
            and row.source_name_exported != source.name
        ) or (
            row.target_name_exported is not None
            and row.target_name_exported != target.name
        ):
            index.name_mismatches.append(
                {
                    "source_file": row.source_file,
                    "line_number": row.line_number,
                    "model_id": (
                        row.source_id
                        if row.source_name_exported != source.name
                        else row.target_id
                    ),
                    "exported_name": (
                        row.source_name_exported
                        if row.source_name_exported != source.name
                        else row.target_name_exported
                    ),
                    "authoritative_name": (
                        source.name
                        if row.source_name_exported != source.name
                        else target.name
                    ),
                }
            )

    for key in sorted(seen):
        canonical = seen[key]
        index.ground_truth.append(canonical)
        index.outgoing.setdefault(canonical.source_id, []).append(canonical)
    # Deterministic ordering of each process's outgoing relationships:
    # subprocesses first, then cross-references, then by target name and ID.
    category_order = {CATEGORY_SUBPROCESS: 0, CATEGORY_CROSS_REFERENCE: 1}
    for source_id, rels in index.outgoing.items():
        rels.sort(
            key=lambda rel: (
                category_order.get(rel.category, 99),
                models_by_id[rel.target_id].name.casefold(),
                rel.target_id,
            )
        )
    return index
