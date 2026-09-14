# Glossary

- grain: what one row in a dataset represents.
- primary key: field(s) that uniquely identify one row.
- foreign key: field linking one table to another table's primary key.
- nullability: whether a field can be missing (`null`).
- invariant: condition that must always hold true.
- FEN: Forsyth-Edwards Notation for board state serialization.
- normalized FEN: canonicalized subset of FEN used for stable position identity.
- position_id: deterministic hash derived from normalization version + normalized FEN.
- pre-move state: board state immediately before the played move.
- leakage: using information unavailable at prediction time.
- legal move mask: fixed-size boolean vector indicating legal action indices.
- fixture: tiny controlled dataset committed for deterministic tests and CI.
- strict mode: ingestion mode that raises on first invalid record.
- tolerant mode: ingestion mode that logs/reports invalid records and continues.
- schema version: explicit version tag for contract compatibility.
- encoding version: explicit version tag for tensor or action mapping compatibility.