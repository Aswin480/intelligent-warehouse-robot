"""Map grid cells (row, col) <-> Gazebo world coordinates (x, y) in meters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridConfig:
    cell_size: float
    origin_x: float
    origin_y: float
    n_rows: int
    n_cols: int


def cell_to_world(row, col, cfg):
    x = cfg.origin_x + (col + 0.5) * cfg.cell_size
    y = cfg.origin_y - (row + 0.5) * cfg.cell_size
    return (x, y)


def world_to_cell(x, y, cfg):
    col = int((x - cfg.origin_x) // cfg.cell_size)
    row = int((cfg.origin_y - y) // cfg.cell_size)
    row = max(0, min(cfg.n_rows - 1, row))
    col = max(0, min(cfg.n_cols - 1, col))
    return (row, col)
