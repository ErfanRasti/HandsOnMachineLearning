# %%
"""This code is related to chapter 2 of the book."""
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
