from sklearn.linear_model import LogisticRegression
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

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test):

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
