from gridworld.grid_coords import GridConfig, cell_to_world, world_to_cell


def test_cell_to_world_centers():
    cfg = GridConfig(cell_size=1.0, origin_x=0.0, origin_y=0.0, n_rows=3, n_cols=3)
    # row 0 is at the top -> world y decreases as row increases.
    assert cell_to_world(0, 0, cfg) == (0.5, -0.5)
    assert cell_to_world(2, 2, cfg) == (2.5, -2.5)


def test_world_to_cell_is_inverse():
    cfg = GridConfig(cell_size=0.5, origin_x=-1.0, origin_y=2.0, n_rows=5, n_cols=4)
    for r in range(5):
        for c in range(4):
            x, y = cell_to_world(r, c, cfg)
            assert world_to_cell(x, y, cfg) == (r, c)


def test_world_to_cell_clamps_out_of_bounds():
    cfg = GridConfig(cell_size=1.0, origin_x=0.0, origin_y=0.0, n_rows=2, n_cols=2)
    assert world_to_cell(-100.0, 100.0, cfg) == (0, 0)
    assert world_to_cell(100.0, -100.0, cfg) == (1, 1)
