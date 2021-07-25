# %%
"""This code is related to chapter 2 of the book."""
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
import urllib
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
# fetch_housing_data()
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

# %%
housing.info()
# %%
housing["ocean_proximity"].value_counts()
# %%
housing.describe()
# %%
housing.hist(bins=50, figsize=(20, 15))
plt.show()
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

for train_index, test_index in split.split(housing, housing["income_cat"]):
    """
    We mention the target variable in the second argument of split function,
    to split the data according to this variable.
    """
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# %%
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# %%
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

# %%
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"] / \
    housing["households"]

# %%
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# %%
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
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
# %%
