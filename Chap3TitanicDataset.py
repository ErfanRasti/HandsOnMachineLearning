# %%
"""In this code we wanna analysis titanic dataset\
and predict whether or not a passenger survived based on some attributes."""

# %% Setup

# Importing modules
import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import sklearn
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

MODEL_PATH = "cache\\TitanicDataset"

PREDICTION_PATH = "predictions\\TitanicDataset"
os.makedirs(PREDICTION_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Save the figure on IMAGES_PATH."""
    path = os.path.join(IMAGES_PATH, fig_id + ".", fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def fit_load(nameOfModel,  path=MODEL_PATH, **kwargs):
    """
    fit_load : Fit the model on the training set\
                or load the .pkl file of the saved model.

    Args:
        nameOfModel (String): Name of the model variable as the string
        path (String, optional): Path of the .pkl file. Defaults to modelPath.

    Returns:
        Model: sklearn fitted model
    """
    os.makedirs(path, exist_ok=True)

    model = eval(nameOfModel)
    filePath = os.path.join(path, nameOfModel+".pkl")

    if os.path.isfile(filePath):

        # loading the model
        model = joblib.load(filePath)

    else:
        model.fit(**kwargs)

        # saving the model
        joblib.dump(model, filePath)

    return model


def cross_val_function(nameOfFile,
                       crossValFunction,
                       path=MODEL_PATH,
                       **kwargs):
    """
    cross_val_function : Evaluate a parameter on a model by cross-validation.

    Args:
        nameOfFile (String): Name of file to save
        crossValFunction (Model): Type of cross-validation model
        path (String, optional): Path of the .pkl file. Defaults to modelPath.

    Returns:
        model: sklearn model using cross-validation
    """
    os.makedirs(path, exist_ok=True)

    filePath = os.path.join(path, nameOfFile+".pkl")

    if os.path.isfile(filePath):

        # loading the model
        model = joblib.load(filePath)

    else:
        model = crossValFunction(**kwargs)

        # saving the model
        joblib.dump(model, filePath)

    return model

# %% Loading the dataset


TITANIC_PATH = os.path.join("datasets", "titanic")


def load_dataset(filename, path=TITANIC_PATH):
    """
    load_dataset : Load the dataset.

    Args:
        filename (<class 'str'>): name dataset file
        path (<class 'str'>, optional): path of dataset file.
            Defaults to TITANIC_PATH.

    Returns:
        <class 'pandas.core.frame.DataFrame'>: dataframe in pandas format
    """
    csv_path = os.path.join(path, filename)

    return pd.read_csv(csv_path)


train_data = load_dataset("train.csv")
test_data = load_dataset("test.csv")

# %% Improving the data

# AgeBucket determines the age range.
train_data["AgeBucket"] = (train_data["Age"] // 15) * 15


# RelativesOnboard
train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]


# %% Preproccessing the dataset


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    """This class determines most frequent categorical values."""

    def fit(self, X, y=None):
        """Fit the dataframe with a new attribute most_frequent_."""
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0]
                                         for c in X],
                                        index=X.columns)
        return self

    def transform(self, X, y=None):
        """Transform the dataframe and\
        fill NA values with most_frequent_ values."""
        return X.fillna(self.most_frequent_)


num_attributes = ["AgeBucket", "RelativesOnboard", "Fare"]
cat_attributes = ["Pclass", "Sex", "Embarked"]


"""
SimpleImputer: Imputation transformer for completing missing values.
    If "median", then replace missing values using the median
    along each column. Can only be used with numeric data.
"""

preprocess_pipeline = ColumnTransformer([
    ("num_pipeline",  SimpleImputer(strategy="median"), num_attributes),
    ("cat_pipeline", Pipeline([
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ]), cat_attributes),
])


X_train = preprocess_pipeline.fit_transform(
    train_data.drop("Survived", axis='columns'))
y_train = train_data["Survived"]


# %% Training a Classifier

# SVC: Support Vector Classifier
svm_clf = SVC(gamma="auto")
svm_clf = fit_load("svm_clf", X=X_train, y=y_train)

svm_scores = cross_val_function("svm_scores",
                                cross_val_score,
                                estimator=svm_clf,
                                X=X_train,
                                y=y_train,
                                cv=10)

print("svm_scores:", svm_scores.mean())


# Random Forest Classifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_function("forest_scores",
                                   cross_val_score,
                                   estimator=forest_clf,
                                   X=X_train,
                                   y=y_train,
                                   cv=10)

print("forest_scores", forest_scores.mean())

# Comparing two models
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
# plt.show()

# %% Optimizing the model

forest_clf = RandomForestClassifier(random_state=42)

param_grid = {"n_estimators": (10, 30, 100),
              "bootstrap": (True, False),
              "max_features": (9, 10, 11),
              }
grid_clf = GridSearchCV(estimator=forest_clf,
                        param_grid=param_grid,
                        verbose=1,
                        cv=5,
                        return_train_score=True,
                        n_jobs=-1)

grid_clf = fit_load("grid_clf", X=X_train, y=y_train)

print("Best Estimator:", grid_clf.best_estimator_)
print("Best Estimator Score:", grid_clf.best_score_)

# %% Testing the model
best_clf = grid_clf.best_estimator_

test_data["AgeBucket"] = (test_data["Age"] // 15) * 15
test_data["RelativesOnboard"] = test_data["SibSp"] + test_data["Parch"]

X_test = preprocess_pipeline.transform(test_data)

y_pred = best_clf.predict(X_test)

final_pred = {"PassengerId": test_data["PassengerId"], "Survived": y_pred}

pd.DataFrame(final_pred).to_csv(
    PREDICTION_PATH+"\\final_pred.csv", index=False)
