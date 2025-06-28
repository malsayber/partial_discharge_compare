from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
import pickle
import os
import re

logger = logging.getLogger(__name__)

def _latest_checkpoint(model_dir: str, model_type: str):
    """Return path to latest checkpoint and epoch number."""
    pattern = re.compile(fr"{model_type}_epoch(\d+)\.json")
    latest_epoch = 0
    latest_path = None
    if not os.path.exists(model_dir):
        return None, 0
    for fname in os.listdir(model_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = os.path.join(model_dir, fname)
    return latest_path, latest_epoch


class _XGBCheckpoint(xgb.callback.TrainingCallback):
    def __init__(self, model_dir: str, model_type: str, interval: int):
        self.model_dir = model_dir
        self.model_type = model_type
        self.interval = interval

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        if (epoch + 1) % self.interval == 0:
            path = os.path.join(
                self.model_dir, f"{self.model_type}_epoch{epoch + 1}.json"
            )
            model.save_model(path)
            logger.info("Saved checkpoint %s", path)
        return False


def train_model(
    X_train,
    y_train,
    model_type: str = "DecisionTree",
    params: dict | None = None,
    cv: int = 5,
    scoring: str = "accuracy",
    model_dir: str = "model",
    checkpoint_interval: int | None = None,
    resume: bool = False,
):
    """Train a model with optional checkpointing and resume support.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and labels.
    model_type : str, optional
        Type of model to train.
    params : dict or None, optional
        Hyperparameters passed to the model constructor.
    cv : int, optional
        Number of cross-validation folds.
    scoring : str, optional
        Metric used for cross-validation scoring.
    model_dir : str, optional
        Directory where models and checkpoints are stored.
    checkpoint_interval : int or None, optional
        Save XGBoost checkpoints every ``checkpoint_interval`` boosting rounds.
    resume : bool, optional
        If ``True``, resume training from the latest checkpoint if available.
    """
    logger.info("Training %s model...", model_type)
    if model_type == 'DecisionTree':
        model = DecisionTreeClassifier(**(params or {}))
    elif model_type == 'SVM':
        model = SVC(**(params or {}), probability=True)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(**(params or {}))
    elif model_type == 'XGBoost':
        model = xgb.XGBClassifier(**(params or {}))
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(**(params or {}))
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    logger.info("Cross-validation scores (%s): %s", scoring, cv_scores)
    logger.info("Mean CV score (%s): %.4f", scoring, cv_scores.mean())

    callbacks = []
    fit_kwargs = {}
    if model_type == "XGBoost" and checkpoint_interval:
        callbacks.append(_XGBCheckpoint(model_dir, model_type, checkpoint_interval))
    if model_type == "XGBoost" and resume:
        ckpt_path, _ = _latest_checkpoint(model_dir, model_type)
        if ckpt_path:
            logger.info("Resuming from checkpoint %s", ckpt_path)
            fit_kwargs["xgb_model"] = ckpt_path
    model.fit(X_train, y_train, **fit_kwargs, callbacks=callbacks)  # Fit after CV

    os.makedirs(model_dir, exist_ok=True) # Ensure model directory exists
    model_path = os.path.join(model_dir, f'{model_type}_model.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    logger.info("Trained %s model saved to %s", model_type, model_path)

    return model, cv_scores.mean()

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    """
    logger.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted') # Use 'weighted' for multi-class
    recall = recall_score(y_test, y_pred, average='weighted')    # Use 'weighted' for multi-class
    f1 = f1_score(y_test, y_pred, average='weighted')        # Use 'weighted' for multi-class
    conf_matrix = confusion_matrix(y_test, y_pred)

    logger.info("Test Accuracy: %.4f", accuracy)
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

    # ROC AUC for binary or multiclass (ovr)
    try:
        y_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr') # 'ovr' for multiclass
        logger.info("Test ROC AUC (OVR): %.4f", roc_auc)
    except AttributeError: # Models without predict_proba (e.g., some SVM kernels without probability=True)
        logger.warning("ROC AUC score not available for this model.")
        roc_auc = None

    logger.info("Test Precision: %.4f", precision)
    logger.info("Test Recall: %.4f", recall)
    logger.info("Test F1-Score: %.4f", f1)
    logger.info("Confusion Matrix:\n%s", conf_matrix)
    logger.info("Evaluation completed.")
    return accuracy, roc_auc, precision, recall, f1, conf_matrix

if __name__ == '__main__':
    from .data_loader import load_data
    from .data_processor import preprocess_data

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, target='target') # Assuming 'target' column exists

    model, cv_mean_score = train_model(X_train, y_train, model_type='RandomForest')
    evaluate_model(model, X_test, y_test)
