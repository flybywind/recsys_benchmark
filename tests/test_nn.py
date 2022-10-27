import torch
import unittest
# must explicitly specify the package `src`
# if u place the `tests` dir inside `src`, and use `import nn` or `import utils` etc
# nn and utils can be taken as the top packages, so if somewhere u wrote `import ..utils`
# you can get error: "ValueError: attempted relative import beyond top-level package"
from src.nn import EmbeddingAll
from src.utils import build_input_features
from src.utils import SparseFeat, DenseFeat


class TestEmbeddingAll(unittest.TestCase):
    def test_embedding_forward(self):
        feature_columns = [SparseFeat('s1', 10, embedding_dim=10), SparseFeat('s2', 6), DenseFeat('d1'), DenseFeat('d2'), DenseFeat('d3')]
        feature_index = build_input_features(feature_columns)
        emb_layer = EmbeddingAll(feature_columns, feature_index, dense_emb_dim=4)
        X = torch.FloatTensor([[1, 2, 0.4, 0.6, 0.8], [9, 2, 0.3, 0.5, 0.7]])
        Y = emb_layer(X)
        self.assertEqual(Y.shape[1], 5, "shape not correct")
        self.assertEqual(Y.shape[2], 4, "shape not correct")
        a = (Y[1, 2, :] / Y[0, 2, :])*4
        a = a.detach().int().numpy()
        self.assertEqual(all(a==3), True)
        a = (Y[0, 1, :] == Y[1, 1, :])
        a = a.detach().int().numpy()
        self.assertEqual(all(a==1), True)
