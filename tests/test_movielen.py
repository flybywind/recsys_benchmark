import torch
import unittest
from src.utils import load_dataset, get_folder_path


class TestMovieLen(unittest.TestCase):
    def test_append_all_attr(self):

        dataset_args = {
            "dataset": "Movielens",
            "dataset_name": "latest-small",
            "if_use_features": False,
            "num_core": 10,
            "num_feat_core": 10,
            "num_negative_samples": 10,
            "cf_loss_type": "BPR",
            "type": "hete",
            "sampling_strategy": "unseen",
            "append_all_entities": True,
            "entity_aware": False}
        data_folder, weights_folder, logger_folder = \
            get_folder_path(model='test', dataset=dataset_args["dataset"] + dataset_args["dataset_name"], loss_type="bpr")
        print(f"folder: {data_folder}")
        dataset_args['root'] = data_folder
        dataset_args['name'] = dataset_args['dataset_name']

        dataset = load_dataset(dataset_args)
        self.assertNotEqual(dataset, None)
        dataset.cf_negative_sampling()
        row = dataset[0]
        print(row)