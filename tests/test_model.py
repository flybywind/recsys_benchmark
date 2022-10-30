import torch
import unittest
from src.models import AFN
from src.nn import EmbeddingAll
from src.utils import build_input_features
from src.utils import SparseFeat, DenseFeat
class TestAFN(unittest.TestCase):
    def test_forward(self):
        feature_columns = [SparseFeat('s1', 10), SparseFeat('s2', 6), DenseFeat('d1'), DenseFeat('d2'), DenseFeat('d3')]
        afn = AFN(feature_columns, ltl_hidden_size=16, afn_dnn_hidden_units=[8, 4], dense_emb_dim=4, task='binary')
        X = torch.FloatTensor([[1, 2, 0.4, 0.6, 0.8], [9, 2, 0.3, 0.5, 0.7]])
        Y = afn(X)
        print(Y)
        print(afn)
        afn2 = AFN(feature_columns, ltl_hidden_size=16, afn_dnn_hidden_units=[8, 4], dense_emb_dim=4, task='binary', l2_reg_shadow=0.001)
        Y2 = afn2(X)
        print(Y2)
