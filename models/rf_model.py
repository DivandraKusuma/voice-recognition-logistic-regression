"""
Random Forest Model untuk Gender Recognition by Voice.

Model ini menggunakan Pipeline dengan StandardScaler agar data
terdistribusi normal sebelum masuk ke classifier.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def train_model(X_train, y_train):
    """
    Melatih model Random Forest dengan StandardScaler pipeline.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Data fitur training.
    y_train : array-like, shape (n_samples,)
        Label target training.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline yang sudah di-fit (scaler + random forest).
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi model pada data test.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Model yang sudah dilatih.
    X_test : array-like, shape (n_samples, n_features)
        Data fitur testing.
    y_test : array-like, shape (n_samples,)
        Label target testing.

    Returns
    -------
    results : dict
        Dictionary berisi accuracy, precision, recall, f1,
        classification_report, dan confusion_matrix.
    """
    y_pred = model.predict(X_test)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', pos_label='male'),
        'recall': recall_score(y_test, y_pred, average='binary', pos_label='male'),
        'f1': f1_score(y_test, y_pred, average='binary', pos_label='male'),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
    }

    return results
