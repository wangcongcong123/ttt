# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# the script template is from: https://github.com/huggingface/datasets/blob/master/templates/new_metric_script.py
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import datasets

_CITATION = """\

"""

_DESCRIPTION = """\
"""

_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
"""

# BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"

def simple_accuracy(preds, labels):
    return {"acc": (np.array(preds) == np.array(labels)).mean()}

def acc_precision_recall_fscore(preds, labels):
    metrics = simple_accuracy(preds, labels)
    macro_precision, macro_recall, macro_fscore, _ = precision_recall_fscore_support(labels, preds, average='macro')
    micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(labels, preds, average='micro')
    weighted_precision, weighted_recall, weighted_fscore, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    metrics.update({"macro_precision": macro_precision, "macro_recall": macro_recall, "macro_fscore": macro_fscore})
    metrics.update({"micro_precision": micro_precision, "micro_recall": micro_recall, "micro_fscore": micro_fscore})
    metrics.update({"weighted_precision": weighted_precision, "weighted_recall": weighted_recall, "weighted_fscore": weighted_fscore})
    return metrics

class ClsMetric(datasets.Metric):
    """customized metric"""

    def _info(self):
        if self.config_name not in [
            "short",
            "long",
        ]:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["short", "long"]'
            )

        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="xxx",
            # Additional links to the codebase or references
            codebase_urls=["xxx"],
            reference_urls=["xxx"]
        )

    # def _download_and_prepare(self, dl_manager):
    #     """Optional: download external resources useful to compute the scores"""
    #     # TODO: Download external resources if needed
    #     bad_words_path = dl_manager.download_and_extract(BAD_WORDS_URL)
    #     self.bad_words = set([w.strip() for w in open(bad_words_path, "r", encoding="utf-8")])
    def _compute(self, predictions, references):
        """Returns the scores"""
        if self.config_name == "short":
            return simple_accuracy(predictions, references)
        elif self.config_name == "long":
            return acc_precision_recall_fscore(predictions, references)
        else:
            raise ValueError(
                "Invalid config name for CLS: {}. Please use 'short' or 'long'.".format(self.config_name))
