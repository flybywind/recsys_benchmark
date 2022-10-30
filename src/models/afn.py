# -*- coding:utf-8 -*-
"""
Author:
    Weiyu Cheng, weiyu_cheng@sjtu.edu.cn

Reference:
    [1] Cheng, W., Shen, Y. and Huang, L. 2020. Adaptive Factorization Network: Learning Adaptive-Order Feature
         Interactions. Proceedings of the AAAI Conference on Artificial Intelligence. 34, 04 (Apr. 2020), 3609-3616.
"""
import torch
import torch.nn as nn

from .base import CtrDNNRecModel
from nn import LogTransformLayer, DNN, ShadowNN


class AFN(CtrDNNRecModel):
    """Instantiates the Adaptive Factorization Network architecture.
    
    In DeepCTR-Torch, we only provide the non-ensembled version of AFN for the consistency of model interfaces. For the ensembled version of AFN+, please refer to https://github.com/WeiyuCheng/DeepCTR-Torch (Pytorch Version) or https://github.com/WeiyuCheng/AFN-AAAI-20 (Tensorflow Version).

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param ltl_hidden_size: integer, the number of logarithmic neurons in AFN
    :param afn_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of DNN layers in AFN
    :param l2_reg_shadow: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.
    
    """

    def __init__(self,
                 feature_columns,
                 ltl_hidden_size=256, afn_dnn_hidden_units=(256, 128),
                 dense_emb_dim=8, max_norm=None,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_shadow=None,
                 init_std=0.0001, seed=1024, dnn_dropout=0, activation='relu',
                 task='ltr', device='cpu', gpus=None, dtype=torch.float32):

        super(AFN, self).__init__(feature_columns, dense_emb_dim=dense_emb_dim, max_norm=max_norm,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus, dtype=dtype)
        field_num = len(feature_columns)

        self.ltl = LogTransformLayer(field_num, dense_emb_dim, ltl_hidden_size)
        self.afn_dnn = DNN(dense_emb_dim * ltl_hidden_size, afn_dnn_hidden_units,
                        activation=activation, dropout_rate=dnn_dropout, use_bn=True, device=device)
        self.afn_dnn_linear = nn.Linear(afn_dnn_hidden_units[-1], 1)

        self.shadow_attached = False
        if l2_reg_shadow is not None:
            self.shadow_nn = ShadowNN(self.embedding_size, activate=activation, device=device, dtype=dtype)
            self.add_regularization_weight(self.shadow_nn.parameters(), l2=l2_reg_shadow)
            self.shadow_attached = True

        self.add_regularization_weight(list(self.afn_dnn.parameters()) + list(self.afn_dnn_linear.parameters()), l2=l2_reg_dnn)
        self.to(device)
    
    def forward(self, X):
        X = self.embedding_input(X)

        ltl_result = self.ltl(X)
        afn_logit = self.afn_dnn(ltl_result)
        logit = self.afn_dnn_linear(afn_logit)

        if self.shadow_attached:
            logit += self.shadow_nn(X)

        y_pred = self.out(logit)
        
        return y_pred


    def predict(self, unids, inids):
        '''
        predict base on user profiles vs id profiles
        :param unids:
        :param inids:
        :return:
        '''
        return self.forward(torch.cat([unids, inids], dim=-1))