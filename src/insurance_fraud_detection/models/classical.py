"""Shared classical model registry for the fraud benchmark."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "SVM",
]


def create_model_registry(random_state: int = 42):
    """Create fresh estimators with one documented protocol for all models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1,
            random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            random_state=random_state),
        # probability=True gives a calibrated probability interface for the
        # same validation-threshold/test-evaluation protocol as other models.
        "SVM": SVC(probability=True, random_state=random_state),
    }


MODELS = create_model_registry()
