"""FastAPI application for the warehouse navigation simulator."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from random import Random
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from async_dp import (
    SolverResult,
    asynchronous_value_iteration,
    compare_algorithms,
    generate_path,
    synchronous_value_iteration,
)
from mdp import WarehouseMDP

ROOT = Path(__file__).parent
DEFAULT_GRID = [
    "S..#......",
    ".#.#..#...",
    ".#....#.#.",
    "...##.....",
    ".......#..",
    "#.#....#.#",
    "....#.....",
    ".#....#...",
    "..#.#...#.",
    ".........G",
]


class GenerateRequest(BaseModel):
    rows: int = Field(default=10, ge=3, le=30)
    cols: int = Field(default=10, ge=3, le=30)
    density: float = Field(default=0.2, ge=0.0, le=0.45)
    seed: int | None = Field(default=None, ge=0)


class PathRequest(BaseModel):
    algorithm: str = Field(default="async")


app = FastAPI(title="Intelligent Warehouse Robot Navigation")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
current_mdp = WarehouseMDP(DEFAULT_GRID)
current_results: dict[str, SolverResult] = {}


def _grid_payload() -> dict[str, Any]:
    return {
        "grid": list(current_mdp.grid),
        "rows": current_mdp.rows,
        "cols": current_mdp.cols,
        "state_count": current_mdp.state_count,
        "start": list(current_mdp.start),
        "goal": list(current_mdp.goal),
    }


def _result_payload(result: SolverResult) -> dict[str, Any]:
    path = generate_path(current_mdp, result.policy)
    return {**result.as_dict(), **path}


def _reachable(rows: int, cols: int, obstacles: set[tuple[int, int]]) -> bool:
    queue = deque([(0, 0)])
    visited = {(0, 0)}
    while queue:
        row, column = queue.popleft()
        if (row, column) == (rows - 1, cols - 1):
            return True
        for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (row + row_delta, column + column_delta)
            if (
                0 <= candidate[0] < rows
                and 0 <= candidate[1] < cols
                and candidate not in obstacles
                and candidate not in visited
            ):
                visited.add(candidate)
                queue.append(candidate)
    return False


def _make_grid(request: GenerateRequest) -> list[str]:
    rng = Random(request.seed)
    start = (0, 0)
    goal = (request.rows - 1, request.cols - 1)
    for _ in range(100):
        obstacles = {
            (row, column)
            for row in range(request.rows)
            for column in range(request.cols)
            if (row, column) not in {start, goal} and rng.random() < request.density
        }
        if _reachable(request.rows, request.cols, obstacles):
            return [
                "".join(
                    "S" if (row, column) == start else
                    "G" if (row, column) == goal else
                    "#" if (row, column) in obstacles else "."
                    for column in range(request.cols)
                )
                for row in range(request.rows)
            ]
    raise HTTPException(status_code=422, detail="Could not generate a solvable warehouse")


def _run(algorithm: str) -> SolverResult:
    if algorithm == "async":
        result = asynchronous_value_iteration(current_mdp)
        current_results["async"] = result
        return result
    if algorithm == "sync":
        result = synchronous_value_iteration(current_mdp)
        current_results["sync"] = result
        return result
    raise HTTPException(status_code=400, detail="algorithm must be 'async' or 'sync'")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    global current_mdp, current_results
    current_mdp = WarehouseMDP(_make_grid(request))
    current_results = {}
    return _grid_payload()


@app.post("/api/run/async-dp")
def run_async() -> dict[str, Any]:
    return {**_grid_payload(), **_result_payload(_run("async"))}


@app.post("/api/run/value-iteration")
def run_sync() -> dict[str, Any]:
    return {**_grid_payload(), **_result_payload(_run("sync"))}


@app.post("/api/path")
def path(request: PathRequest) -> dict[str, Any]:
    result = current_results.get(request.algorithm) or _run(request.algorithm)
    return {**_grid_payload(), **_result_payload(result)}


@app.post("/api/reset")
def reset() -> dict[str, Any]:
    global current_mdp, current_results
    current_mdp = WarehouseMDP(DEFAULT_GRID)
    current_results = {}
    return _grid_payload()


@app.get("/api/compare")
def compare() -> dict[str, Any]:
    global current_results
    current_results = compare_algorithms(current_mdp)
    return {
        **_grid_payload(),
        "results": {
            name: _result_payload(result)
            for name, result in current_results.items()
        },
    }