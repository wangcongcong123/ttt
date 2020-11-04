from __future__ import absolute_import, division, print_function
import json

import datasets

_CITATION = """\

"""
_DESCRIPTION = """\
"""

_TRAIN_DOWNLOAD_URL = f"data/covid_info/train.json"
_VAL_DOWNLOAD_URL = f"data/covid_info/val.json"


class CovidDataConfig(datasets.BuilderConfig):
    def __init__(
            self,
            **kwargs,
    ):
        # self.second_choice=kwargs.pop("second_choice",None)
        super(CovidDataConfig, self).__init__(version=datasets.Version("0.0.0", ""), **kwargs)


class CovidData(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CovidDataConfig(
            name="default",
            description="",
        ),
    ]
    """customize dataset."""
    # VERSION = datasets.Version("0.0.0")
    def _info(self):
        data_info = datasets.DatasetInfo(
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
        return data_info

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
                    "source": data["text"],
                    "target": data["label"],
                }
