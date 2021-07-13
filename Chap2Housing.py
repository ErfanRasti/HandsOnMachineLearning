# %%
"""This code is related to chapter 2 of the book."""
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
