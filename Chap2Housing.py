"""This code is related to chapter 2 of the book."""

# %%
# Importing Libraries
from scipy.stats import expon, loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.tree import plot_tree
import matplotlib.image as mpimg
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from zlib import crc32
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tarfile
import urllib.request
import joblib

# %% [markdown]
# # Get the Data
# ## Download the Data

# %%

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Fetch the data to the housing_path.

    housing_url -- url of the housing data
    housing_path -- an special path to save the data
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Calling the function fetch_housing_data
fetch_housing_data()


# %%

# loading the data


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Load the dataset using pandas library.

    Args:
        housing_path ([string]): Defaults to HOUSING_PATH.

    Returns:
        <class 'function'>:
        This function returns a pandas DataFrame object
        containing all the data.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# %%
housing = load_housing_data()
housing.head()

# %% [markdown]
# ## Take a Quick Look at the Data Structure

# %%
housing.info()


# %%
housing["ocean_proximity"].value_counts()


# %%
housing.describe()


# %%
housing.hist(bins=50, figsize=(20, 15))
plt.show()

# %% [markdown]
# ## Create a Test Set

# %%


def split_train_test(data, test_ratio):
    """
    Split test data from the dataset.

    Args:
        data([DataFrame]): The main dataset
            containing train data and test data

        test_ratio([float]): the ratio of the test data

    Returns:
        train set([DataFrame])
        test set([DataFrame])


    This method randomizes  an indices array with numpy random function.
    The length of array is the same as length of data.
    Then it calculates test set size and splits indices array
    to train indices and test indices.
    Finally it returns the train set and test set.
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# calling the split_train_test function
train_set, test_set = split_train_test(housing, 0.2)


# %%
len(train_set)


# %%
len(test_set)


# %%

def test_set_check(identifier, test_ratio):
    """Check the identifier with maximum hash value."""
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


# %%

def split_train_test_by_id(data, test_ratio, id_column):
    """Split the data by id_column."""
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# %%

housing_with_id = housing.reset_index()  # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# %%

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# %%
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# %%

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()


# %%

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# using "income_cat" attribute to split the train and test set
for train_index, test_index in split.split(housing, housing["income_cat"]):
    """
    We mention the target variable in the second argument of split function,
    to split the data according to this variable.
    """
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# %%
# removing the redundant attribute
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# %% [markdown]
# # Discover and Visualize the Data to Gain Insights
# ## Visualizing Geographical Data

# %%
# replacing the housing variable with stratified train set
housing = strat_train_set.copy()


# %%
housing.plot(kind="scatter", x="longitude", y="latitude")


# %%
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# %%
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
plt.legend()

# %% [markdown]
# ## Looking for Correlations

# %%
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# %%
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# %%
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)

# %% [markdown]
# ## Experimenting with Attribute Combinations

# %%
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"] / \
    housing["households"]


# %%
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# %% [markdown]
# # Prepare the Data for Machine Learning Algorithms

# %%
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# %% [markdown]
# ## Data Cleaning

# %%
housing.dropna(subset=["total_bedrooms"])  # option 1
housing.drop("total_bedrooms", axis=1)  # option 2
median = housing["total_bedrooms"].median()  # option 3
housing["total_bedrooms"].fillna(median, inplace=True)


# %%
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)


# %%
imputer.statistics_, housing_num.median().values


# %%
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# %% [markdown]
# ## Handling Text and Categorical Attributes

# %%
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# %%
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# %%
ordinal_encoder.categories_


# %%
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# %%
housing_cat_1hot.toarray()


# %%
cat_encoder.categories_

# %% [markdown]
# ## Custom Transformers

# %%

# Index Location
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """This class adds some extra attributes to the dataset."""

    """
    BaseEstimator: Base class for all estimators in scikit-learn.
    TransformerMixin: Mixin class for all transformers in scikit-learn.
    """

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        """Initialize the add_bedrooms_per_room parameter, to determine\
        if this column adds or not."""
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        """Return the dataset itself."""
        return self  # nothing else to do

    def transform(self, X):
        """Transform the dataset with adding some new attributes."""
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            """
            numpy.c_ :
            Translates slice objects to concatenation along the second axis.
            """
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# Defining the object
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# Applying the object on the housing.values and saving it in a new variable
housing_extra_attribs = attr_adder.transform(housing.values)

# %% [markdown]
# ## Feature Scaling
# ## Transformation Pipelines

# %%
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# %%
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

# %% [markdown]
# # Select and Train a Model
# ## Training and Evaluating on the Training Set

# %%
housing_prepared, housing_labels


# %%
cachePath = os.path.join("cache", "Chap2")
os.makedirs(cachePath, exist_ok=True)


# %%

lin_reg = LinearRegression()

if os.path.isfile("cache/Chap2/lin_reg.pkl"):
    # loading the model
    lin_reg = joblib.load("cache/Chap2/lin_reg.pkl")

else:
    lin_reg.fit(housing_prepared, housing_labels)

    # saving the model
    joblib.dump(lin_reg, "cache/Chap2/lin_reg.pkl")


# %%
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))


# %%
print("Labels:", list(some_labels))


# %%
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# %%

tree_reg = DecisionTreeRegressor()

if os.path.isfile("cache/Chap2/tree_reg.pkl"):
    # loading the model
    tree_reg = joblib.load("cache/Chap2/tree_reg.pkl")

else:
    tree_reg.fit(housing_prepared, housing_labels)

    # saving the model
    joblib.dump(tree_reg, "cache/Chap2/tree_reg.pkl")


# %%
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# %% [markdown]
# ## Better Evaluation Using Cross-Validation

# %%


if os.path.isfile("cache/Chap2/tree_scores.pkl"):
    # loading the model
    tree_scores = joblib.load("cache/Chap2/tree_scores.pkl")

else:
    tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                                  scoring="neg_mean_squared_error", cv=10)

    # saving the model
    joblib.dump(tree_scores, "cache/Chap2/tree_scores.pkl")


tree_rmse_scores = np.sqrt(-tree_scores)


# %%
def display_scores(scores):
    """Display the scores."""
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


display_scores(tree_rmse_scores)


# %%

if os.path.isfile("cache/Chap2/lin_scores.pkl"):
    # loading the model
    lin_scores = joblib.load("cache/Chap2/lin_scores.pkl")

else:
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)

    # saving the model
    joblib.dump(lin_scores, "cache/Chap2/lin_scores.pkl")

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# %%

forest_reg = RandomForestRegressor()

if os.path.isfile("cache/Chap2/forest_reg.pkl"):
    # loading the model
    forest_reg = joblib.load("cache/Chap2/forest_reg.pkl")


else:
    forest_reg.fit(housing_prepared, housing_labels)

    # saving the model
    joblib.dump(forest_reg, "cache/Chap2/forest_reg.pkl")

housing_predictions = forest_reg.predict(housing_prepared)


# %%

forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# %%
if os.path.isfile("cache/Chap2/forest_scores.pkl"):
    # loading the model
    forest_scores = joblib.load("cache/Chap2/forest_scores.pkl")

else:
    forest_scores = cross_val_score(forest_reg,
                                    housing_prepared,
                                    housing_labels,
                                    scoring="neg_mean_squared_error",
                                    cv=10)

    # savinge the model
    joblib.dump(forest_scores, "cache/Chap2/forest_scores.pkl")


forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# %% [markdown]
# # Fine-Tune Your Model
# ## Grid Search

# %%

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

if os.path.isfile("cache/Chap2/grid_search.pkl"):
    # loading the model
    grid_search = joblib.load("cache/Chap2/grid_search.pkl")

else:
    grid_search.fit(housing_prepared, housing_labels)

    # saving the model
    joblib.dump(grid_search, "cache/Chap2/grid_search.pkl")


# %%
grid_search.best_params_


# %%
grid_search.best_estimator_


# %%
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# %% [markdown]
# ## Randomized Search
# ## Ensemble Methods
# ## Analyze the Best Models and Their Errors

# %%
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# %%
# hhold: household
# pop: population
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
full_pipeline


# %%
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_encoder


# %%
cat_one_hot_attribs = list(cat_encoder.categories_[0])
cat_one_hot_attribs


# %%
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
attributes


# %%
sorted(zip(feature_importances, attributes), reverse=True)

# %% [markdown]
# ## Evaluate Your System on the Test Set

# %%
final_model = grid_search.best_estimator_
final_model


# %%
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)


# %%
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)

# RMSE: Root Mean Square Error
final_rmse = np.sqrt(final_mse)  # => evaluates to 47,730.2
final_rmse


# %%
# calculating confidence interval
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
"""
df: degree of freedom

scipy.stats.t.interval:
    Endpoints of the range that contains
    fraction alpha [0, 1] of the distribution

squared_errors.mean(): mean_squared_error(y_test, final_predictions)

scipy.stats.sem: Calculate the standard error of the mean
                (or standard error of measurement) of the values
                in the input array.
"""

np.sqrt(stats.t.interval(confidence,
                         len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


# %%

acc = final_model.score(X_test_prepared, y_test)
acc


# %%

fig = plt.figure(figsize=(15, 10))

imagesPath = os.path.join("images", "Chap2")
os.makedirs(imagesPath, exist_ok=True)

if os.path.isfile("images/Chap2/final_model_estimators_0.png"):
    # Reading the image

    img = mpimg.imread("images/Chap2/final_model_estimators_0.png")
    plt.imshow(img)
    plt.axis('off')
else:
    plot_tree(final_model.estimators_[0],
              filled=True, impurity=True,
              rounded=True)

    # Saving the image
    fig.savefig("images/Chap2/final_model_estimators_0.png")

# %% [markdown]
# # Exercises
#
# %% [markdown]
# ## Ex1

# %%


param_grid_SVR = [
    {'kernel': ['linear'], 'C': [30000.0, 100000.0, 300000.0]},
    {'kernel': ['rbf'], 'C': [100.0, 300.0, 1000.0],
     'gamma': [0.1, 0.3, 1.0]},
]

svm_reg = SVR()

"""
verbose: int
    Controls the verbosity: the higher, the more messages.
"""
grid_search_SVR = GridSearchCV(svm_reg,
                               param_grid_SVR,
                               cv=5,
                               scoring='neg_mean_squared_error',
                               verbose=2,
                               n_jobs=-1,
                               return_train_score=True)

if os.path.isfile("cache/Chap2/grid_search_SVR.pkl"):
    # loading the model
    grid_search_SVR = joblib.load("cache/Chap2/grid_search_SVR.pkl")

else:
    grid_search_SVR.fit(housing_prepared, housing_labels)

    # saving the model
    joblib.dump(grid_search_SVR, "cache/Chap2/grid_search_SVR.pkl")


# %%
negative_mse = grid_search_SVR.best_score_
rmse = np.sqrt(-negative_mse)
rmse


# %%
grid_search_SVR.best_params_

# %% [markdown]
# ## Ex2

# %%

"""
see https://docs.scipy.org/doc/scipy/reference/stats.html
for `expon()` and `reciprocal()` documentation
    and more probability distribution functions.

`loguniform()` acts like `reciprocal()`.
"""
# Note: gamma is ignored when kernel is "linear"

param_distribs = {
    'kernel': ['linear', 'rbf'],
    'C': loguniform(20, 200000),
    'gamma': expon(scale=1.0),
}

svm_reg = SVR()
rnd_search_SVR = RandomizedSearchCV(svm_reg,
                                    param_distributions=param_distribs,
                                    n_iter=10,
                                    cv=5,
                                    scoring='neg_mean_squared_error',
                                    verbose=2,
                                    random_state=42,
                                    n_jobs=-1)


if os.path.isfile("cache/Chap2/rnd_search_SVR.pkl"):
    # loading the model
    rnd_search_SVR = joblib.load("cache/Chap2/rnd_search_SVR.pkl")

else:
    rnd_search_SVR.fit(housing_prepared, housing_labels)

    # saving the model
    joblib.dump(rnd_search_SVR, "cache/Chap2/rnd_search_SVR.pkl")


# %%
negative_mse = rnd_search_SVR.best_score_
rmse = np.sqrt(-negative_mse)
rmse


# %%
rnd_search_SVR.best_params_


# %%
expon_distrib = expon(scale=1.)
samples = expon_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Exponential distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()


# %%
# reciprocal.pdf(x, a, b) = 1 / (x*log(b/a))
loguniform_distrib = loguniform(20, 200000)

"""
rvs(a, b, loc=0, scale=1, size=1, random_state=None)

    Random variates.
"""
samples = loguniform_distrib.rvs(10000, random_state=42)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("loguniform distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()

"""The reciprocal distribution is useful\
when you have no idea what the scale of the hyperparameter should be\
(indeed, as you can see on the figure on the right, all scales are\
equally likely, within the given range), whereas the exponential distribution\
is best when you know(more or less) what\
the scale of the hyperparameter should be.
"""
# ## Ex3

# %%


def indices_of_top_k(arr, k):
    """Return the indices of the top k values."""
    return np.sort(np.argsort(np.array(arr))[-k:])
# def indices_of_top_k(arr, k):
#     return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    """This class adds feature indices of top k values to the fitted data."""

    def __init__(self, feature_importances, k):
        """Initialize the class with feature_importance and k value."""
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        """Add feature_indices_ attribute to the fitted data."""
        # self.feature_indices_: indices of top k values
        self.feature_indices_ = indices_of_top_k(
            self.feature_importances, self.k)
        return self

    def transform(self, X):
        """Return the feature_indices_ column."""
        return X[:, self.feature_indices_]


# %%
feature_importances


# %%
k = 5
top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices


# %%
attributes


# %%
np.array(attributes)[top_k_feature_indices]


# %%
sorted(zip(feature_importances, attributes), reverse=True)[:k]


# %%
full_pipeline, TopFeatureSelector(feature_importances, k)


# %%
preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])


# %%
housing_prepared_top_k_features = \
    preparation_and_feature_selection_pipeline.fit_transform(housing)
housing_prepared_top_k_features


# %%
housing_prepared[0:3]


# %%
housing_prepared_top_k_features[0:3]


# %%
housing_prepared[0:3, top_k_feature_indices]

# %% [markdown]
# ## Ex4

# %%

param_distribs_forest = {
    'n_estimators': np.arange(start=10, stop=500, step=10),
    'max_features': np.arange(start=1, stop=6, step=1),
}

forest_reg = RandomForestRegressor()

rnd_search_forest = \
    RandomizedSearchCV(forest_reg,
                       cv=5,
                       param_distributions=param_distribs_forest,
                       n_iter=10,
                       verbose=2,
                       n_jobs=-1,
                       scoring='neg_mean_squared_error',
                       return_train_score=True)

if os.path.isfile("cache/Chap2/rnd_search_forest.pkl"):
    # loading the model
    rnd_search_forest = joblib.load("cache/Chap2/rnd_search_forest.pkl")

else:
    rnd_search_forest.fit(housing_prepared_top_k_features, housing_labels)

    # saving the model
    joblib.dump(rnd_search_forest, "cache/Chap2/rnd_search_forest.pkl")


# %%
rnd_search_forest.best_params_


# %%
cvres_forest = rnd_search_forest.cv_results_
for mean_score, params in \
        zip(cvres_forest["mean_test_score"], cvres_forest["params"]):
    print(np.sqrt(-mean_score), params)


# %%
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('forest_reg', RandomForestRegressor(**rnd_search_forest.best_params_))
])


if os.path.isfile("cache/Chap2/prepare_select_and_predict_pipeline.pkl"):
    # loading the model
    prepare_select_and_predict_pipeline = joblib.load(
        "cache/Chap2/prepare_select_and_predict_pipeline.pkl")

else:
    prepare_select_and_predict_pipeline.fit(housing, housing_labels)

    # saving the model
    joblib.dump(prepare_select_and_predict_pipeline,
                "cache/Chap2/prepare_select_and_predict_pipeline.pkl")


# %%
some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))


# %%
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

prediction_rnd_forest_full_pipeline = \
    prepare_select_and_predict_pipeline.predict(X_test)

final_mse = mean_squared_error(y_test, prediction_rnd_forest_full_pipeline)
final_rmse = np.sqrt(final_mse)
final_rmse

# %% [markdown]
# ## Ex5

# %%
param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline,
                                param_grid,
                                cv=5,
                                scoring='neg_mean_squared_error',
                                verbose=2)


if os.path.isfile("cache/Chap2/grid_search_prep.pkl"):
    # loading the model
    grid_search_prep = joblib.load("cache/Chap2/grid_search_prep.pkl")

else:
    grid_search_prep.fit(housing, housing_labels)

    # saving the model
    joblib.dump(grid_search_prep, "cache/Chap2/grid_search_prep.pkl")


# %%
grid_search_prep.best_params_


# %%
some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", grid_search_prep.predict(some_data))
print("Labels:\t\t", list(some_labels))


# %%
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

prediction_grid_search_prep = grid_search_prep.predict(X_test)

final_mse = mean_squared_error(y_test, prediction_grid_search_prep)
final_rmse = np.sqrt(final_mse)
final_rmse


# %%
grid_search_prep.best_estimator_
