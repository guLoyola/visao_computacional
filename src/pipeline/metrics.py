import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    classification_report,
    roc_auc_score,
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize


class ClassificationMetrics:
    def __init__(self, y_true, y_pred, y_pred_proba, class_names):
        self.y_true: np.ndarray = np.array(y_true)
        self.y_pred: np.ndarray = np.array(y_pred)
        self.y_pred_proba: np.ndarray = np.array(y_pred_proba)
        self.class_names = class_names
        self.n_classes: int = len(class_names)
        self.accuracy: float = 0.0
        self.precision_weighted: float = 0.0
        self.recall_weighted: float = 0.0
        self.f1_weighted: float = 0.0
        self.precision_per_class: np.ndarray = np.array([])
        self.recall_per_class: np.ndarray = np.array([])
        self.f1_per_class: np.ndarray = np.array([])
        self.support_per_class: np.ndarray = np.array([])
        self.conf_matrix: np.ndarray = np.array([])
        self.mcc: float = 0.0
        self.kappa: float = 0.0
        self.roc_auc_weighted: float | None = None
        self.roc_auc_per_class: np.ndarray | None = None
        self.specificity_per_class: np.ndarray = np.array([])
        self.classification_report_str: str = ""

        self._calculate_all_metrics()

    def _calculate_all_metrics(self):
        self.accuracy = accuracy_score(self.y_true, self.y_pred)

        precision_w, recall_w, f1_w, support_w = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='weighted'
        )
        self.precision_weighted = precision_w
        self.recall_weighted = recall_w
        self.f1_weighted = f1_w

        precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None, labels=list(range(self.n_classes))
        )
        self.precision_per_class = np.array(precision_c)
        self.recall_per_class = np.array(recall_c)
        self.f1_per_class = np.array(f1_c)
        self.support_per_class = np.array(support_c)

        self.conf_matrix = confusion_matrix(self.y_true, self.y_pred)

        self.mcc = matthews_corrcoef(self.y_true, self.y_pred)

        self.kappa = cohen_kappa_score(self.y_true, self.y_pred)

        try:
            y_true_bin = label_binarize(
                self.y_true, classes=list(range(self.n_classes)))
            self.roc_auc_weighted = float(roc_auc_score(
                y_true_bin, self.y_pred_proba, multi_class='ovr', average='weighted'
            ))
            roc_per_class = roc_auc_score(
                y_true_bin, self.y_pred_proba, multi_class='ovr', average=None  # type: ignore
            )
            self.roc_auc_per_class = np.array(
                roc_per_class) if roc_per_class is not None else None
        except Exception as e:
            print(f"Warning: Could not calculate ROC AUC: {e}")
            self.roc_auc_weighted = None
            self.roc_auc_per_class = None

        self.specificity_per_class = self._calculate_specificity()

        report = classification_report(
            self.y_true, self.y_pred, target_names=self.class_names, digits=4, output_dict=False
        )
        self.classification_report_str = str(report) if report else ""

    def _calculate_specificity(self):
        specificity = []
        for i in range(self.n_classes):
            # True Negatives: samples that are not class i and predicted as not class i
            tn = np.sum((self.y_true != i) & (self.y_pred != i))
            # False Positives: samples that are not class i but predicted as class i
            fp = np.sum((self.y_true != i) & (self.y_pred == i))
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity.append(spec)
        return np.array(specificity)

    def get_summary_dict(self):
        return {
            'accuracy': self.accuracy,
            'precision_weighted': self.precision_weighted,
            'recall_weighted': self.recall_weighted,
            'f1_weighted': self.f1_weighted,
            'mcc': self.mcc,
            'kappa': self.kappa,
            'roc_auc_weighted': self.roc_auc_weighted,
            'confusion_matrix': self.conf_matrix
        }

    def get_per_class_dict(self):
        per_class = {}
        for i, class_name in enumerate(self.class_names):
            per_class[class_name] = {
                'precision': float(self.precision_per_class[i]),
                'recall': float(self.recall_per_class[i]),
                'f1_score': float(self.f1_per_class[i]),
                'specificity': float(self.specificity_per_class[i]),
                'support': int(self.support_per_class[i])
            }
            if self.roc_auc_per_class is not None:
                per_class[class_name]['roc_auc'] = float(
                    self.roc_auc_per_class[i])
        return per_class

    def print_summary(self):
        """Imprime um resumo formatado das métricas"""
        print("\n" + "=" * 70)
        print("RESUMO DAS MÉTRICAS DE AVALIAÇÃO")
        print("=" * 70)

        print("\n--- MÉTRICAS GERAIS ---")
        print(f"Accuracy                       : {self.accuracy:.4f}")
        print(
            f"Precision (weighted)           : {self.precision_weighted:.4f}")
        print(f"Recall (weighted)              : {self.recall_weighted:.4f}")
        print(f"F1-Score (weighted)            : {self.f1_weighted:.4f}")
        print(
            f"Matthews Correlation Coef.     : {self.mcc:.4f}")
        print(f"Cohen's Kappa                  : {self.kappa:.4f}")
        if self.roc_auc_weighted is not None:
            print(
                f"ROC AUC (weighted)             : {self.roc_auc_weighted:.4f}")

        print("\n--- MÉTRICAS POR CLASSE ---")
        per_class = self.get_per_class_dict()
        for class_name, metrics in per_class.items():
            print(f"\n{class_name}:")
            print(
                f"  Precision   : {metrics['precision']:.4f}  |  Recall      : {metrics['recall']:.4f}")
            print(
                f"  F1-Score    : {metrics['f1_score']:.4f}  |  Specificity : {metrics['specificity']:.4f}")
            if 'roc_auc' in metrics:
                print(
                    f"  ROC AUC     : {metrics['roc_auc']:.4f}  |  Support     : {metrics['support']}")
            else:
                print(f"  Support     : {metrics['support']}")

        print("\n--- CLASSIFICATION REPORT ---")
        print(self.classification_report_str)

        print("=" * 70 + "\n")

    def save_to_txt(self, filepath):
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("RESUMO DAS MÉTRICAS DE AVALIAÇÃO\n")
            f.write("=" * 70 + "\n\n")

            f.write("--- MÉTRICAS GERAIS ---\n")
            f.write(f"Accuracy                       : {self.accuracy:.4f}\n")
            f.write(
                f"Precision (weighted)           : {self.precision_weighted:.4f}\n")
            f.write(
                f"Recall (weighted)              : {self.recall_weighted:.4f}\n")
            f.write(
                f"F1-Score (weighted)            : {self.f1_weighted:.4f}\n")
            f.write(
                f"Matthews Correlation Coef.     : {self.mcc:.4f}\n")
            f.write(f"Cohen's Kappa                  : {self.kappa:.4f}\n")
            if self.roc_auc_weighted is not None:
                f.write(
                    f"ROC AUC (weighted)             : {self.roc_auc_weighted:.4f}\n")

            f.write("\n--- MÉTRICAS POR CLASSE ---\n")
            per_class = self.get_per_class_dict()
            for class_name, metrics in per_class.items():
                f.write(f"\n{class_name}:\n")
                f.write(
                    f"  Precision   : {metrics['precision']:.4f}  |  Recall      : {metrics['recall']:.4f}\n")
                f.write(
                    f"  F1-Score    : {metrics['f1_score']:.4f}  |  Specificity : {metrics['specificity']:.4f}\n")
                if 'roc_auc' in metrics:
                    f.write(
                        f"  ROC AUC     : {metrics['roc_auc']:.4f}  |  Support     : {metrics['support']}\n")
                else:
                    f.write(f"  Support     : {metrics['support']}\n")

            f.write("\n--- CLASSIFICATION REPORT ---\n")
            f.write(self.classification_report_str)
            f.write("\n" + "=" * 70 + "\n")

        print(f"✓ Métricas salvas em: {filepath}")


def calculate_metrics(y_true, y_pred, y_pred_proba, class_names):
    return ClassificationMetrics(y_true, y_pred, y_pred_proba, class_names)


def compare_models(metrics_dict):
    comparison = {
        'model_names': list(metrics_dict.keys()),
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mcc': [],
        'roc_auc': []
    }

    for _, metrics_obj in metrics_dict.items():
        comparison['accuracy'].append(metrics_obj.accuracy)
        comparison['precision'].append(metrics_obj.precision_weighted)
        comparison['recall'].append(metrics_obj.recall_weighted)
        comparison['f1'].append(metrics_obj.f1_weighted)
        comparison['mcc'].append(metrics_obj.mcc)
        if metrics_obj.roc_auc_weighted is not None:
            comparison['roc_auc'].append(metrics_obj.roc_auc_weighted)
        else:
            comparison['roc_auc'].append(0.0)

    return comparison
