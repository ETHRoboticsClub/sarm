
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

from sarm.dataset.gap_dataset import GapLerobotDataset


def test_gap_dataset():
    repo_id = 'ETHRC/piper_towel_v0'
    
    dataset_meta = LeRobotDatasetMetadata(repo_id)
    action_horizon = 25
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in  ("action",)
        },
    )
    # Test Action Chunk
    dataset_gab = GapLerobotDataset(repo_id=repo_id,action_horizon=25, frame_gap=30, t_step_lookback=8)
    np.testing.assert_array_equal(dataset[124]['action'][0], dataset[100]['action'][24])
    assert dataset[0]['action'].shape == (action_horizon, dataset_meta.shapes['action'][0])
    np.testing.assert_array_equal(dataset_gab[100]['action'], dataset[100]['action'])
    
    # Test Gap Data
    assert 'gab_data_0.action' in dataset_gab[0]
    assert 'gab_data_1.action' in dataset_gab[0]
