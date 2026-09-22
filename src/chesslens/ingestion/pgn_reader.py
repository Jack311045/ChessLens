"""Streaming PGN reader for compressed Lichess archives with record extraction."""

from __future__ import annotations

import hashlib
import hmac
import io
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chess
import chess.pgn
import zstandard

from chesslens.domain.records import (
    BOARD_ENCODING_VERSION,
    SCHEMA_VERSION,
    GameRecord,
    IngestionErrorRecord,
    MoveRecord,
    PositionRecord,
    Recoverability,
)
from chesslens.features.position_encoding import (
    POSITION_NORMALIZATION_VERSION,
    extract_fen_metadata,
    position_id_from_normalized_fen,
)

_MONTH_PATTERN = re.compile(r"(\d{4}-\d{2})")
_CLOCK_PATTERN = re.compile(r"\[%clk\s+([^\]]+)\]")
_COMPLETE_RESULTS = {"1-0", "0-1", "1/2-1/2"}

PlayerHashMode = Literal["fixture_placeholder", "hmac_sha256"]


@dataclass(frozen=True)
class ParsedGame:
    game_record: GameRecord
    move_records: list[MoveRecord]
    position_records: list[PositionRecord]


class GameParseError(Exception):
    def __init__(self, error_record: IngestionErrorRecord) -> None:
        super().__init__(error_record.error_message)
        self.error_record = error_record


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def infer_source_month(source_archive_name: str) -> str | None:
    match = _MONTH_PATTERN.search(source_archive_name)
    if match is None:
        return None
    return match.group(1)


def stable_game_id(source_archive_sha256: str, source_game_index: int) -> str:
    payload = f"{source_archive_sha256}|{source_game_index}".encode()
    return hashlib.sha256(payload).hexdigest()


def hash_player_identifier(
    player_name: str | None,
    secret_key: str | None = None,
    mode: PlayerHashMode = "fixture_placeholder",
) -> str | None:
    """
    Return stable anonymized player identifiers.

    In production this should use explicit HMAC mode with a secret key.
    Fixture mode provides deterministic placeholder hashing for sanitized local data.
    """
    if player_name is None:
        return None
    normalized = player_name.strip().lower()
    if not normalized:
        return None

    raw = normalized.encode("utf-8")
    if mode == "hmac_sha256":
        if not secret_key:
            raise ValueError("hmac_sha256 mode requires a non-empty secret key")
        digest = hmac.new(secret_key.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    elif mode == "fixture_placeholder":
        digest = hashlib.sha256(f"fixture-placeholder|{normalized}".encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported player hash mode: {mode}")
    return f"player_{digest[:24]}"


def extract_clock_annotation(comment: str | None) -> str | None:
    if not comment:
        return None
    match = _CLOCK_PATTERN.search(comment)
    if match is None:
        return None
    return match.group(1)


def _parse_int_header(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned == "?":
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _build_error(
    *,
    source_archive: str,
    source_game_index: int,
    error_code: str,
    error_message: str,
    run_id: str,
    recoverability: Recoverability = "recoverable",
) -> IngestionErrorRecord:
    return IngestionErrorRecord(
        source_archive=source_archive,
        source_game_index=source_game_index,
        error_code=error_code,
        error_message=error_message,
        handling_decision="rejected",
        recoverability=recoverability,
        run_id=run_id,
    )


def iter_raw_games(
    archive_path: Path,
    max_games: int | None = None,
) -> Iterator[tuple[int, chess.pgn.Game]]:
    """Stream games from a .pgn.zst archive one-by-one without loading the full file."""
    if max_games is not None and max_games < 0:
        raise ValueError("max_games must be non-negative")

    with archive_path.open("rb") as compressed_stream:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(compressed_stream) as binary_stream:
            with io.TextIOWrapper(binary_stream, encoding="utf-8", errors="replace") as text_stream:
                game_index = 0
                while max_games is None or game_index < max_games:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    yield game_index, game
                    game_index += 1


def parse_game_to_records(
    *,
    game: chess.pgn.Game,
    source_archive: str,
    source_archive_sha256: str,
    source_game_index: int,
    schema_version: str = SCHEMA_VERSION,
    run_id: str = "local-dev",
    player_hash_mode: PlayerHashMode = "fixture_placeholder",
    player_hash_secret: str | None = None,
) -> ParsedGame:
    if game.errors:
        message = "; ".join(str(err) for err in game.errors)
        raise GameParseError(
            _build_error(
                source_archive=source_archive,
                source_game_index=source_game_index,
                error_code="pgn_parse_error",
                error_message=message,
                run_id=run_id,
            )
        )

    variant = game.headers.get("Variant")
    if variant and variant not in {"Standard", "Chess"}:
        raise GameParseError(
            _build_error(
                source_archive=source_archive,
                source_game_index=source_game_index,
                error_code="unsupported_variant",
                error_message=f"Unsupported variant: {variant}",
                run_id=run_id,
            )
        )

    game_id = stable_game_id(source_archive_sha256, source_game_index)
    try:
        board = game.board()
    except ValueError as exc:
        raise GameParseError(
            _build_error(
                source_archive=source_archive,
                source_game_index=source_game_index,
                error_code="invalid_fen",
                error_message=f"invalid fen: {exc}",
                run_id=run_id,
            )
        ) from exc

    move_records: list[MoveRecord] = []
    position_records_by_id: dict[str, PositionRecord] = {}

    for ply, node in enumerate(game.mainline()):
        move = node.move
        if move not in board.legal_moves:
            raise GameParseError(
                _build_error(
                    source_archive=source_archive,
                    source_game_index=source_game_index,
                    error_code="illegal_move",
                    error_message=f"Illegal move at ply {ply}: {move.uci()}",
                    run_id=run_id,
                )
            )

        pre_move_fen = board.fen(en_passant="legal")
        metadata = extract_fen_metadata(pre_move_fen)
        position_id = position_id_from_normalized_fen(
            metadata.normalized_fen,
            normalization_version=POSITION_NORMALIZATION_VERSION,
        )

        if position_id not in position_records_by_id:
            position_records_by_id[position_id] = PositionRecord(
                position_id=position_id,
                normalized_fen=metadata.normalized_fen,
                side_to_move=metadata.side_to_move,
                castling_rights=metadata.castling_rights,
                en_passant_square=metadata.en_passant_square,
                encoding_version=BOARD_ENCODING_VERSION,
                schema_version=schema_version,
            )

        move_records.append(
            MoveRecord(
                game_id=game_id,
                ply=ply,
                position_id=position_id,
                pre_move_fen=pre_move_fen,
                normalized_pre_move_fen=metadata.normalized_fen,
                side_to_move=metadata.side_to_move,
                played_move_uci=move.uci(),
                played_move_san=board.san(move),
                halfmove_clock=metadata.halfmove_clock,
                fullmove_number=metadata.fullmove_number,
                clock_annotation_raw=extract_clock_annotation(node.comment),
                schema_version=schema_version,
            )
        )

        board.push(move)

    if not move_records:
        raise GameParseError(
            _build_error(
                source_archive=source_archive,
                source_game_index=source_game_index,
                error_code="empty_game",
                error_message="Game has no mainline moves",
                run_id=run_id,
            )
        )

    game_record = GameRecord(
        game_id=game_id,
        source_archive=source_archive,
        source_archive_sha256=source_archive_sha256,
        source_month=infer_source_month(source_archive),
        source_game_index=source_game_index,
        event=game.headers.get("Event"),
        played_date=game.headers.get("Date"),
        round=game.headers.get("Round"),
        result=game.headers.get("Result"),
        white_rating=_parse_int_header(game.headers.get("WhiteElo")),
        black_rating=_parse_int_header(game.headers.get("BlackElo")),
        time_control_raw=game.headers.get("TimeControl"),
        eco=game.headers.get("ECO"),
        opening=game.headers.get("Opening"),
        termination=game.headers.get("Termination"),
        white_player_hash=hash_player_identifier(
            game.headers.get("White"),
            secret_key=player_hash_secret,
            mode=player_hash_mode,
        ),
        black_player_hash=hash_player_identifier(
            game.headers.get("Black"),
            secret_key=player_hash_secret,
            mode=player_hash_mode,
        ),
        ply_count=len(move_records),
        schema_version=schema_version,
    )

    return ParsedGame(
        game_record=game_record,
        move_records=move_records,
        position_records=list(position_records_by_id.values()),
    )


def iter_parsed_games(
    archive_path: Path,
    *,
    max_games: int | None,
    strict: bool,
    schema_version: str = SCHEMA_VERSION,
    source_archive_sha256: str | None = None,
    run_id: str = "local-dev",
    on_error: Callable[[IngestionErrorRecord], None] | None = None,
    player_hash_mode: PlayerHashMode = "fixture_placeholder",
    player_hash_secret: str | None = None,
) -> Iterator[ParsedGame]:
    source_archive = archive_path.name
    archive_sha256 = source_archive_sha256 or compute_sha256(archive_path)

    for source_game_index, game in iter_raw_games(archive_path=archive_path, max_games=max_games):
        try:
            yield parse_game_to_records(
                game=game,
                source_archive=source_archive,
                source_archive_sha256=archive_sha256,
                source_game_index=source_game_index,
                schema_version=schema_version,
                run_id=run_id,
                player_hash_mode=player_hash_mode,
                player_hash_secret=player_hash_secret,
            )
        except GameParseError as exc:
            if on_error:
                on_error(exc.error_record)
            if strict:
                raise


def is_complete_game(game_record: GameRecord) -> bool:
    return game_record.result in _COMPLETE_RESULTS and game_record.ply_count > 0