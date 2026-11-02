# NORA: modified version of nlu_metric.py

"""NLU Metric"""

import datasets
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
import pandas as pd
import sys
import argparse


# TODO: Add BibTeX citation
_CITATION = """\
"""

_DESCRIPTION = """\
Metric to evaluate text-to-text models on the natural language understanding task.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: list of predictions to score. Each predictions
        should be a string.
    references: list of reference for each prediction. Each
        reference should be a string.
Returns:
    accuracy
    classification report
    2 classes F1 (custom_F1)
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NLUMetrics(datasets.Metric):
    """Metric to evaluate text-to-text models on the natural language understanding task."""

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            })
        )

    def _compute(self, predictions, references):

        #### !!! CHANGE THIS !!!! ######
        file_name = "zero_All"
        file_dir = "output/zero_All/eval/"
        #######################

        print("Size of refrences:", len(references), ",Size of predections:", len(predictions))

        with open(file_dir + '_gold.txt', 'w') as f:
            for line in references:
                f.write('{}\n'.format(line))

        with open(file_dir + '_pred.txt', 'w') as f:
            for line in predictions:
                f.write('{}\n'.format(line))
        
        acc = []
        for prediction, reference in zip(predictions, references):
            acc.append(prediction==reference)

        report_test = classification_report(references, predictions, zero_division=0, digits=4, output_dict=True)
        print(report_test)
        df_report_test = pd.DataFrame(report_test).transpose()
        custom_f1 = two_classes_f1(references, predictions)
        df_report_test.loc['Two classes F1'] = custom_f1
        ## Generate CSV for all metrics
        df_report_test.to_csv(file_dir + file_name + '_report.csv', sep=',')


        return {
            "accuracy": sum(acc)/len(acc),
            "Macro_f1": f1_score(references, predictions, average='macro'),
            "Micro_f1": f1_score(references, predictions, average='micro'), ## Micro is the same as accuracy
            "Weighted_f1": f1_score(references, predictions, average='weighted'),
            "two_classes_f1": two_classes_f1(references, predictions)
            #f1_sarcastic = f1_score(truths,submitted, average = "binary", pos_label = 1) -->used in sarcasm detection
        }
    
    

def two_classes_f1(references, predictions):
    # Create a mapping dictionary
    label_map = {
        'مؤيد': 0,
        'محايد': 1,
        'معارض': 2,
        'in favor': 0,
        'neutral': 1,
        'against': 2,
        'ضد': 2,
        'في favor': 0,
        ##########
        'إيجابي': 0,
        'سلبي': 2,
        'نعم': 1,
        'لا': 1,
        '': 1,
        'positive':0,
        'negative': 2
    }
    
    # Map the references and predictions to 0/1/2
    references_mapped = [label_map[label] for label in references]
    predictions_mapped = [label_map[label] for label in predictions]
    
    # Get precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(references_mapped, predictions_mapped, labels=[0, 1, 2])
    
    # Calculate the customized F1 score as the average of the F1 score for favor and against classes
    custom_f1 = (f1[0] + f1[2]) / 2

    return custom_f1



