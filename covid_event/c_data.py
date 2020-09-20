# coding=utf-8
# This script is finished following HF's datasets' template:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
# More examples as references to write a customized dataset can be found here:
# https://github.com/huggingface/datasets/tree/master/datasets

from __future__ import absolute_import, division, print_function

import json

import datasets

_CITATION = """\

"""
_DESCRIPTION = """\
"""

_TRAIN_DOWNLOAD_URL = "data/final/train.json"
_VAL_DOWNLOAD_URL = "data/final/val.json"

class CData(datasets.GeneratorBasedBuilder):
    """covid event data script."""
    # VERSION = datasets.Version("1.0.0")
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "source": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="#",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        val_path = dl_manager.download_and_extract(_VAL_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": val_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                yield id_, {
                    "source": data["source"],
                    "target": data["target"],
                }
