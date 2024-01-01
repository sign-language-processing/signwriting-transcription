from plistlib import Dict
from typing import List
from tokenizer.signwriting import SignSymbol
from base import SignWritingMetric


class SymbolsDistancesMetric(SignWritingMetric):
    def __init__(self):
        super().__init__()

    def compere_signs_score(self, pred: SignSymbol, gold: SignSymbol) -> float:
        """
        Calculate the compere signs score for a given prediction and gold.
        :param pred: the prediction
        :param gold: the gold
        :return: the FWS score
        """
        pass

    def penalty_error(self, pred) -> float:
        """
        Calculate the penalty error for a given prediction.
        :param pred: the prediction
        :return: the penalty error
        """
        pass

    def fws_score(self, pred) -> float:
        """
        Calculate the FWS score for a given prediction.
        :param pred: the prediction
        :return: the FWS score
        """
        pass

    def pred_signs(self, pred) -> List[SignSymbol]:
        """
        Get the signs from a prediction.
        :param pred: the prediction
        :return: the signs
        """
        pass

    def error_rate(self, pred, gold) -> float:
        """
        Calculate the evaluate score for a given prediction and gold.
        :param pred: the prediction
        :param gold: the gold
        :return: the FWS score
        """
        pass

    def score(self, pred, gold) -> float:
        """
        Calculate the evaluate score for a given prediction and gold.
        :param pred: the prediction
        :param gold: the gold
        :return: the FWS score
        """
        1 - self.error_rate(pred, gold)
