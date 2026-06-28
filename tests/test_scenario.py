from gridworld.scenario import load_scenario


def test_load_maze_builds_mdp():
    sc = load_scenario("scenarios/maze.yaml")
    mdp = sc.build_mdp()
    assert mdp.n_states > 0
    assert mdp.shape == (sc.coords.n_rows, sc.coords.n_cols)
    # The start cell must be a navigable (non-wall) state.
    assert sc.start in mdp.index
