import numpy as np
from legal_retrieval.indexing.dense import NumpyFlatIndex


def test_numpy_flat_dense_backend(tmp_path):
    index = NumpyFlatIndex({})
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype='float32')
    index.build(embeddings, ['a', 'b'])
    hits = index.search(np.array([0.9, 0.1], dtype='float32'), 2)
    assert hits[0].record_id == 'a'
    index.save(tmp_path)
    loaded = NumpyFlatIndex.load(tmp_path, {})
    assert loaded.search(np.array([0.1, 0.9], dtype='float32'), 1)[0].record_id == 'b'
