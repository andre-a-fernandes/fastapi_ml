from typing import Literal, Optional

from joblib import dump
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def load_dataset(dataset_name: Literal["iris", "other"]):
    if dataset_name != "iris":
        raise NotImplementedError()
    
    dataset = datasets.load_iris(return_X_y=True)
    return dataset[0], dataset[1]

def train_ml_classifier(X, y, output_file: Optional[str] = None):
    clf_pipeline = [('scaling', MinMaxScaler()), 
                    ('classifier', DecisionTreeClassifier(random_state=42))]
    pipeline = Pipeline(clf_pipeline)
    pipeline.fit(X, y)

    if output_file is not None:
        dump(pipeline, output_file)


if __name__ == '__main__':

    X, y = load_dataset('iris')
    model = train_ml_classifier(X, y, output_file='./iris_v1.joblib')
