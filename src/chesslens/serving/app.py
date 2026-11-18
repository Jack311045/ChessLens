"""Typed single-position API; no training or audit data is read during inference."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

import chess
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chesslens.features.action_encoding import decode_index_to_move, legal_move_mask
from chesslens.modeling.labels import rating_band, time_control_category
from chesslens.modeling.neural_data import encode_position
from chesslens.modeling.onnx_parity import (
    OUTPUT_SPEC,
    rank_legal_actions,
    softmax_wdl,
    validate_inputs,
)
from chesslens.serving.bundle import ServingBundle, load_serving_bundle, validate_outputs

Rating = Annotated[int, Field(strict=True, ge=0, le=10000)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PlayerContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mover_rating: Rating | None = None
    opponent_rating: Rating | None = None
    time_control: Annotated[str, Field(max_length=64)] | None = None


class RankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fen: Annotated[str, Field(min_length=1, max_length=256)]
    context: PlayerContext | None = None
    top_k: Annotated[int, Field(strict=True, ge=1, le=218)] = 5

    @field_validator("fen")
    @classmethod
    def valid_fen(cls, fen: str) -> str:
        parts = fen.split()
        if len(parts) != 6:
            raise ValueError("FEN must contain all six fields, including move counters")
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError(
                f"FEN is not a valid standard chess position (status={board.status()})"
            )
        if not 1 <= int(parts[5]) <= 2147483647 or board.halfmove_clock > 2147483647:
            raise ValueError("FEN counters must fit nonnegative int32; fullmove must be positive")
        return fen


class HealthResponse(BaseModel):
    status: Literal["ready"]


class ModelInfo(BaseModel):
    model_version: str
    model_sha256: str
    preprocessing_sha256: str
    modeling_dataset_id: str
    bundle_schema_version: str
    encoding_versions: dict[str, str]
    runtime_provider: Literal["CPUExecutionProvider"]
    wdl_class_order: tuple[Literal["win"], Literal["draw"], Literal["loss"]]
    temperature: Annotated[float, Field(ge=1, le=1)]
    intra_op_threads: int
    inter_op_threads: int


class RankedMove(BaseModel):
    rank: int
    uci: str
    action_index: int
    probability: Probability


class WDL(BaseModel):
    win: Probability
    draw: Probability
    loss: Probability


class RankResponse(BaseModel):
    status: Literal["ok", "terminal"]
    terminal_reason: str | None
    side_to_move: Literal["white", "black"]
    legal_move_count: int
    moves: list[RankedMove]
    wdl: WDL | None
    model_version: str
    model_sha256: str
    processing_time_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]


def encode_request(request: RankRequest, bundle: ServingBundle) -> dict[str, np.ndarray]:
    board = chess.Board(request.fen)
    context = request.context or PlayerContext()
    mover, opponent = context.mover_rating, context.opponent_rating
    features = encode_position(
        request.fen,
        numeric_context={
            "mover_rating": mover,
            "opponent_rating": opponent,
            "rating_difference": None if mover is None or opponent is None else mover - opponent,
            "ply": board.ply(),
        },
        mover_rating_band=rating_band(mover),
        time_control_category=time_control_category(context.time_control),
        preprocessor=bundle.preprocessor,
    )
    inputs: dict[str, np.ndarray] = {
        "board_tensor": features.board_tensor[None],
        "continuous_context": features.continuous_context[None],
        "rating_band_index": np.asarray([features.rating_band_index], dtype=np.int64),
        "time_control_index": np.asarray([features.time_control_index], dtype=np.int64),
    }
    validate_inputs(inputs, bundle.preprocessor)
    return inputs


def rank_position(request: RankRequest, bundle: ServingBundle) -> RankResponse:
    start = perf_counter()
    board = chess.Board(request.fen)
    outcome = board.outcome(claim_draw=False)
    moves: list[RankedMove] = []
    wdl = None
    legal_count = board.legal_moves.count()
    if outcome is None:
        inputs = encode_request(request, bundle)
        policy, value = validate_outputs(bundle.session.run(list(OUTPUT_SPEC), inputs))
        ranked = rank_legal_actions(policy[0], legal_move_mask(board), request.fen)
        scores = policy[0, ranked].astype(np.float64)
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        moves = [
            RankedMove(
                rank=rank + 1,
                uci=decode_index_to_move(action, board).uci(),
                action_index=action,
                probability=float(probabilities[rank]),
            )
            for rank, action in enumerate(ranked[: request.top_k])
        ]
        win, draw, loss = softmax_wdl(value)[0]
        wdl = WDL(win=float(win), draw=float(draw), loss=float(loss))
    return RankResponse(
        status="terminal" if outcome is not None else "ok",
        terminal_reason=outcome.termination.name.lower() if outcome is not None else None,
        side_to_move="white" if board.turn else "black",
        legal_move_count=legal_count,
        moves=moves,
        wdl=wdl,
        model_version=bundle.metadata.experiment_id,
        model_sha256=bundle.model_sha256,
        processing_time_ms=(perf_counter() - start) * 1000,
    )


def create_app(bundle_path: Path | None = None, *, threads: int = 2) -> FastAPI:
    bundle: ServingBundle | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal bundle
        configured = bundle_path or os.environ.get("CHESSLENS_BUNDLE_DIR")
        if configured is None:
            raise RuntimeError(
                "Set CHESSLENS_BUNDLE_DIR or pass --bundle to configure the ONNX bundle"
            )
        bundle = load_serving_bundle(Path(configured), threads=threads)
        try:
            yield
        finally:
            bundle = None

    app = FastAPI(title="ChessLens single-position inference", version="1", lifespan=lifespan)

    def ready() -> ServingBundle:
        if bundle is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return bundle

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ready()
        return HealthResponse(status="ready")

    @app.get("/v1/model-info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        loaded = ready()
        return ModelInfo(
            model_version=loaded.metadata.experiment_id,
            model_sha256=loaded.model_sha256,
            preprocessing_sha256=loaded.preprocessing_sha256,
            modeling_dataset_id=loaded.metadata.modeling_dataset_id,
            bundle_schema_version=loaded.metadata.schema_version,
            encoding_versions=loaded.metadata.versions,
            runtime_provider="CPUExecutionProvider",
            wdl_class_order=("win", "draw", "loss"),
            temperature=1.0,
            intra_op_threads=loaded.session.get_session_options().intra_op_num_threads,
            inter_op_threads=loaded.session.get_session_options().inter_op_num_threads,
        )

    @app.post("/v1/rank-moves", response_model=RankResponse)
    def rank_moves(request: RankRequest) -> RankResponse:
        return rank_position(request, ready())

    return app
