import numpy as np
from gridworld.scenario import load_scenario
from viz.value_animation import value_iteration_frames
from viz.compare import compare_vi_pi


def test_frames_align_with_value_iteration_sweeps():
    sc = load_scenario("scenarios/maze.yaml")
    mdp = sc.build_mdp()
    frames, final_V, final_policy = value_iteration_frames(mdp)
    assert len(frames) >= 1
    assert frames[0].shape == mdp.shape
    # The last frame equals the converged value function on the grid.
    assert np.allclose(np.nan_to_num(frames[-1]),
                       np.nan_to_num(mdp.grid_values(final_V)))
    assert final_policy.shape == (mdp.n_states, mdp.n_actions)


def test_compare_reports_agreement():
    sc = load_scenario("scenarios/maze.yaml")
    result = compare_vi_pi(sc)
    assert result["agree"] is True
    assert result["vi_sweeps"] >= 1
    assert result["pi_iters"] >= 1
