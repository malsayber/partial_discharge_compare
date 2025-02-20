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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(X_train, y_train, model_type='DecisionTree', params=None, cv=5, scoring='accuracy', model_dir='model'):
    """
    Trains and evaluates a machine learning model using cross-validation.
    Saves the trained model to model_dir.
    """
    logging.info(f"Training {model_type} model...")
    if model_type == 'DecisionTree':
        model = DecisionTreeClassifier(**(params or {})) # Use params if provided, else default
    elif model_type == 'SVM':
        model = SVC(**(params or {}), probability=True) # probability=True for ROC AUC
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(**(params or {}))
    elif model_type == 'XGBoost':
        model = xgb.XGBClassifier(**(params or {}))
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(**(params or {}))
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    logging.info(f"Cross-validation scores ({scoring}): {cv_scores}")
    logging.info(f"Mean CV score ({scoring}): {cv_scores.mean():.4f}")

    model.fit(X_train, y_train) # Fit on the entire training set after CV

    os.makedirs(model_dir, exist_ok=True) # Ensure model directory exists
    model_path = os.path.join(model_dir, f'{model_type}_model.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f"Trained {model_type} model saved to {model_path}")

    return model, cv_scores.mean()

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    """
    logging.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted') # Use 'weighted' for multi-class
    recall = recall_score(y_test, y_pred, average='weighted')    # Use 'weighted' for multi-class
    f1 = f1_score(y_test, y_pred, average='weighted')        # Use 'weighted' for multi-class
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # ROC AUC for binary or multiclass (ovr)
    try:
        y_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr') # 'ovr' for multiclass
        logging.info(f"Test ROC AUC (OVR): {roc_auc:.4f}")
    except AttributeError: # Models without predict_proba (e.g., some SVM kernels without probability=True)
        logging.warning("ROC AUC score not available for this model.")
        roc_auc = None

    logging.info(f"Test Precision: {precision:.4f}")
    logging.info(f"Test Recall: {recall:.4f}")
    logging.info(f"Test F1-Score: {f1:.4f}")
    logging.info("Confusion Matrix:\n" + str(conf_matrix))
    logging.info("Evaluation completed.")
    return accuracy, roc_auc, precision, recall, f1, conf_matrix

if __name__ == '__main__':
    from data_loader import load_data
    from data_processor import preprocess_data

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, target='target') # Assuming 'target' column exists

    model, cv_mean_score = train_model(X_train, y_train, model_type='RandomForest')
    evaluate_model(model, X_test, y_test)
