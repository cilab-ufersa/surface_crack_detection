import keras
import sys


class Metrics:
    def __init__(self, args):
        self.args = args

    def define_Metrics(self):

        sys.path.append(self.args["main"])

        from subroutines.loss_metrics import Recall
        from subroutines.loss_metrics import Precision
        from subroutines.loss_metrics import Precision_dil
        from subroutines.loss_metrics import F1_score
        from subroutines.loss_metrics import F1_score_dil

        # Prepare metrics
        metrics = [
            keras.metrics.BinaryAccuracy(name='accuracy'),
            Recall,
            Precision,
            Precision_dil,
            F1_score,
            F1_score_dil
        ]

        return metrics
