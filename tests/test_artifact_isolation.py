from copy import deepcopy

from legal_retrieval.config import load_config
from legal_retrieval.indexing.manager import IndexManager


def test_embedding_models_have_distinct_immutable_index_directories():
    first = load_config("configs/base.yaml")
    first["indexing"]["dense"]["model_name"] = "model/a"
    second = deepcopy(first)
    second["indexing"]["dense"]["model_name"] = "model/b"

    first_manager = IndexManager(first)
    second_manager = IndexManager(second)

    assert first_manager.index_root == second_manager.index_root
    assert first_manager.index_signature != second_manager.index_signature
    assert first_manager.index_dir != second_manager.index_dir


def test_identical_index_configs_share_the_same_artifact():
    cfg = load_config("configs/base.yaml")
    assert IndexManager(cfg).index_dir == IndexManager(deepcopy(cfg)).index_dir
