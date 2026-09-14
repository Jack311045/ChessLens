# Encoding Specification (Phase 0)

## Prediction Point and Leakage Guard

A training row represents state immediately before the played move.

- pre-move board state: feature candidate.
- played move: label.
- game result and termination: labels/metadata only.
- post-move clock annotations and future opening labels: prohibited as pre-move features.

Ply convention is zero-based:

- `ply = 0`: White first move.
- capture FEN before `board.push(move)`.

## Position Normalization and Identity

Normalization version: `fen4_legal_ep_v1`.

Rules:

1. Canonicalize with `python-chess` legal en-passant representation.
2. Keep first four FEN fields only:
   - piece placement
   - side to move
   - castling rights
   - en-passant target (`-` if absent)
3. Keep halfmove/fullmove separately when needed for metadata.

Position ID:

- `position_id = sha256(normalization_version + "|" + normalized_fen)`.

Important limitation:

- normalized FEN does not encode full repetition history; it is not a complete game-history identity.

## Board Encoding

Encoding version: `board18_abs_v1`.

Shape and layout:

- channel-first `(18, 8, 8)`.
- absolute orientation:
  - file index `0` -> `a`
  - rank index `0` -> `1`
  - indexing: `[channel, rank, file]`
- dtype: `float32`.

Channel order:

0. white pawns
1. white knights
2. white bishops
3. white rooks
4. white queens
5. white king
6. black pawns
7. black knights
8. black bishops
9. black rooks
10. black queens
11. black king
12. side to move plane (all ones if White to move, otherwise zeros)
13. white king-side castling right plane (all ones/zeros)
14. white queen-side castling right plane (all ones/zeros)
15. black king-side castling right plane (all ones/zeros)
16. black queen-side castling right plane (all ones/zeros)
17. legal en-passant target one-hot square plane

Invariants:

- deterministic repeated calls;
- function does not mutate board state.

## Action Encoding

Encoding version: `action8x8x73_v1`.

Global action space:

- `8 x 8 x 73 = 4672` actions.
- index formula: `action_index = from_square * 73 + move_plane`.
- `from_square` uses `python-chess` square numbering.

Plane groups:

- planes `0..55`: sliding moves
  - 8 directions x 7 distances
- planes `56..63`: knight offsets
- planes `64..72`: underpromotions
  - 3 relative directions (`left`, `forward`, `right`) x 3 promoted pieces (`N`, `B`, `R`)

Special move handling:

- queen promotion uses corresponding sliding plane.
- castling uses king two-square movement.
- en passant uses pawn diagonal movement.

Decoding requires board context for promotion resolution.

## Legal Mask

`legal_move_mask(board)` returns shape `(4672,)` boolean mask.

- `True`: legal move for this position.
- `False`: illegal/unavailable move.

Invariants:

- mask true-count equals number of legal moves.
- no collisions among legal moves in same position.