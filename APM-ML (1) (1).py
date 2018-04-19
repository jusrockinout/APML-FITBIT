
# coding: utf-8

# ***Get The Data!***

# In[4]:


# Setting Up Function to Fetch CSV Data File

import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Function Call to Fetch CSV Data File

fetch_housing_data()


# In[5]:


#Setting Up Function to Load Fetched CSV Data File

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# ***Explore The Data!***

# In[6]:


# Function Call to Load Fetched CSV Data File

housing = load_housing_data()
housing.head()


# In[7]:


# Method Call info() to Generate Description of Data

housing.info()


# In[8]:


# Method Call describe() to Generate Summary of Numerical Attributes

housing.describe()

# Count is 
# Mean is 
# Std is 
# min
# 25%
# 50%
# 75%
# max
 


# In[9]:


# Method Call hist() to Generate Summary of Numerical Attributes: the number of instances (on the vertical axis) that have a given value range (on the horizontal axis).

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[10]:


# CREATE A TEST SET!!!!

# Setting Up Function to Split A Test Set


import numpy as np


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# Function Call to Execute Function to Split Test Set from Data

train_set, test_set = split_train_test(housing, 0.2)

print(len(train_set), "train +", len(test_set), "test")


# In[11]:


# This ensures that the test set will remain consistent across multiple runs, even if you refresh the dataset. The new test set will contain 20% of the new instances, but it will not contain any instance that was previously in the training set.

import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[12]:


# adds an `index` column

housing_with_id = housing.reset_index()

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[13]:


# If you use the row index as a unique identifier, you need to make sure that new data gets appended to the end of the dataset, and
# no row ever gets deleted. If this is not possible, then you can try to use the most stable features to build a unique identifier. For
# example, a district’s latitude and longitude are guaranteed to be stable for a few million years, so you could combine them into
# an ID like so:

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[14]:


test_set.head()


# In[15]:


# Scikit-Learn provides a few functions to split datasets into multiple subsets in various ways. The simplest function is 
# train_test_split, which does pretty much the same thing as the function split_train_test defined earlier, with a couple of additional features

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[16]:


test_set.head()


# In[17]:


housing["median_income"].hist()


# In[18]:


# Stratification for Generalization

# The following code creates an income category attribute by dividing the median income by 1.5 (to limit the number of income
# categories), and rounding up using ceil (to have discrete categories), and then merging all the categories greater than 5 into category 5

# Divide by 1.5 to limit the number of income categories

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

# Label those above 5 as 5

housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[19]:


housing["income_cat"].value_counts()


# In[20]:


housing["income_cat"].hist()


# In[21]:


# Now you are ready to do stratified sampling based on the income category. For this you can use Scikit-Learn’s StratifiedShuffleSplit class:

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Verification of Stratified Test Set

# Let’s see if this worked as expected. You can start by looking at the income category proportions in the full housing dataset:
    
housing["income_cat"].value_counts() / len(housing)


# In[22]:


# Stratified Test Set for Comparison

strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[23]:


# Compares the income category
# proportions in the overall dataset, in the test set generated with stratified sampling, and in a test set generated using purely
# random sampling. As you can see, the test set generated using stratified sampling has income category proportions almost
# identical to those in the full dataset, whereas the test set generated using purely random sampling is quite skewed.

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100


# In[24]:


# Runs Aforementioned Function

compare_props


# In[25]:


# Now you should remove the income_cat attribute so the data is back to its original state

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# ***Explore The Data: Visually!***

# In[26]:


# Create Copy Before Continuing Data Exploration

housing = strat_train_set.copy()


# In[27]:


# Geographical Distribution

housing.plot(kind="scatter", x="longitude", y="latitude")


# In[28]:


# Better Visualization

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[29]:


# Even Better Visualization

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()


# In[30]:


# Looking for Correlations


# Linear Correlations

corr_matrix = housing.corr()

# How much each attribute correlates with the median house value:

corr_matrix["median_house_value"].sort_values(ascending=False)



# In[31]:


# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
# Not 11x11 Features is too many!

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[32]:


# Zoom in

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])


# 
# ***Experimenting with Attribute Combinations***
# 

# In[33]:


# Creating New Features: Ratios of Others

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[34]:


# Rerunning Feature Correlation Due to Updated Features

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[35]:


#  The new bedrooms_per_room attribute is much more correlated with the median house value than the total
# number of rooms or bedrooms. Apparently houses with a lower bedroom/room ratio tend to be more expensive. The number of
# rooms per household is also more informative than the total number of rooms in a district — obviously the larger the houses,
# the more expensive they are.


# In[36]:


# Scatter Plot of Rooms/Household vs. Median House Value

housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()


# In[37]:


# Rerunning describe() Method to Generate Updated Summary of Numerical Attributes

housing.describe()


# 
# ***Preparing the Data!***
# 

# In[38]:


# But first let’s revert to a clean training set (by copying strat_train_set once again), and let’s separate the predictors and the
# labels since we don’t necessarily want to apply the same transformations to the predictors and the target values (note that
# drop() creates a copy of the data and does not affect strat_train_set)

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


# **Data Cleaning!**

# In[39]:


# Most Machine Learning algorithms cannot work with missing features, so let’s create a few functions to take care of them.

# Three Options

# Get rid of the corresponding districts.
# Get rid of the whole attribute.
# Set the values to some value (zero, the mean, the median, etc.).

# You can accomplish these easily using DataFrame’s dropna(), drop(), and fillna() methods:

# housing.dropna(subset=["total_bedrooms"]) # option 1
# housing.drop("total_bedrooms", axis=1) # option 2
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median) # option 3



# In[40]:


sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[41]:


sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1


# In[42]:


sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2


# In[43]:


median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
sample_incomplete_rows


# In[44]:


# Scikit-Learn provides a handy class to take care of missing values: Imputer. Here is how to use it. First, you need to create an
# Imputer instance, specifying that you want to replace each attribute’s missing values with the median of that attribute:

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")


# In[45]:


# Remove the text attribute because median can only be calculated on numerical attributes:

housing_num = housing.drop('ocean_proximity', axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])


# In[46]:


# Now you can fit the imputer instance to the training data using the fit() method:

imputer.fit(housing_num)


# In[47]:


# The imputer has simply computed the median of each attribute and stored the result in its statistics_ instance variable.

imputer.statistics_


# In[48]:


# Check that this is the same as manually computing the median of each attribute:
housing_num.median().values


# In[49]:


# Transform the training set:
# The result is a plain Numpy array containing the transformed features.

X = imputer.transform(housing_num)


# In[50]:


# s. If you want to put it back into a Pandas DataFrame, it’s simple:

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index = list(housing.index.values))


# In[51]:


housing_tr.loc[sample_incomplete_rows.index.values]


# In[52]:


imputer.strategy


# In[53]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.head()


# ***Handling Text and Categorical Attributes!***
# 

# In[54]:


# Now let's preprocess the categorical input feature, ocean_proximity

housing_cat = housing['ocean_proximity']
housing_cat.head(10)


# In[55]:


# We can use Pandas' factorize() method to convert this string categorical feature to an integer categorical feature, 
# which will be easier for Machine Learning algorithms to handle:


housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]


# In[56]:


housing_categories


# In[57]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot


# In[58]:


# The OneHotEncoder returns a sparse array by default, but we can convert it to a dense array if needed

housing_cat_1hot.toarray()


# In[59]:


# Definition of the CategoricalEncoder class, copied from PR #9151.
# Just run this cell, or copy it to your code, do not try to understand it (yet).

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """
    
    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


# In[60]:


# The CategoricalEncoder expects a 2D array containing one or more categorical input features. We need to reshape housing_cat to a 2D array:

#from sklearn.preprocessing import CategoricalEncoder # in future versions of Scikit-Learn

cat_encoder = CategoricalEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot


# In[61]:


# The default encoding is one-hot, and it returns a sparse array. You can use toarray() to get a dense array:

housing_cat_1hot.toarray()



# In[62]:


# Alternatively, you can specify the encoding to be "onehot-dense" to get a dense matrix rather than a sparse matrix:

cat_encoder = CategoricalEncoder(encoding="onehot-dense")
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot


# In[63]:


cat_encoder.categories_


# ***Custom Transformers***
# 

# In[64]:


# Let's create a custom transformer to add extra attributes:

from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# In this example the transformer has one hyperparameter, add_bedrooms_per_room, set to True by default (it is often helpful to
# provide sensible defaults). This hyperparameter will allow you to easily find out whether adding this attribute helps the
# Machine Learning algorithms or not. More generally, you can add a hyperparameter to gate any data preparation step that you
# are not 100% sure about. The more you automate these data preparation steps, the more combinations you can automatically try
# out, making it much more likely that you will find a great combination (and saving you a lot of time).


# In[65]:


housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()


# ***Feature Scaling***

# ***Transformation Pipelines*** 
# 

# In[66]:


# Now let's build a pipeline for preprocessing the numerical attributes:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[67]:


housing_num_tr


# In[68]:


# And a transformer to just select a subset of the Pandas DataFrame columns:

from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[69]:


# Now let's join all these components into a big pipeline that will preprocess both the numerical and the categorical features:

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])


# In[70]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[71]:


housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


# In[72]:


housing_prepared.shape


# ***Select And Train a Model***

# ***Training and Evaluating on the Training Set***
# 

# In[73]:


# The good news is that thanks to all these previous steps, things are now going to be much simpler than you might think. Let’s
# first train a Linear Regression model

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Done! You now have a working Linear Regression model. Let’s try it out on a few instances from the training set:


# In[81]:


# let's try the full pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

# Compare against the actual values:

print("Labels:", list(some_labels))

# It works, although the predictions are not exactly accurate (e.g., the second prediction is off by more than 50%!).


# In[76]:


some_data_prepared


# In[83]:


# Let’s measure this regression model’s RMSE on the whole training set using Scikit-Learn’s mean_squared_error function:

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


# Okay, this is better than nothing but clearly not a great score: most districts’ median_housing_values range between
# $120,000 and $265,000, so a typical prediction error of $68,628 is not very satisfying. This is an example of a model
# underfitting the training data. When this happens it can mean that the features do not provide enough information to make good
# predictions, or that the model is not powerful enough. As we saw in the previous chapter, the main ways to fix underfitting are
# to select a more powerful model, to feed the training algorithm with better features, or to reduce the constraints on the model.
# This model is not regularized, so this rules out the last option. You could try to add more features (e.g., the log of the
# population), but first let’s try a more complex model to see how it does.


# In[78]:


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae


# In[84]:


# Let’s train a DecisionTreeRegressor. This is a powerful model, capable of finding complex nonlinear relationships in the data

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# In[85]:


# Now that the model is trained, let’s evaluate it on the training set:

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[86]:


# Wait, what!? No error at all? Could this model really be absolutely perfect? Of course, it is much more likely that the model
# has badly overfit the data. How can you be sure? As we saw earlier, you don’t want to touch the test set until you are ready to
# launch a model you are confident about, so you need to use part of the training set for training, and part for model validation.


# ***Better Evaluation Using Cross-Validation***
# 

# In[88]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[89]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# In[90]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[91]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[92]:


housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[93]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[94]:


scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()


# In[95]:


from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse


# ***Fine Tune Model***

# ***Grid Search***

# In[98]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[99]:


grid_search.best_params_


# In[100]:


grid_search.best_estimator_


# In[101]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[ ]:





# In[102]:


pd.DataFrame(grid_search.cv_results_)


# In[103]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)


# In[ ]:


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# ***Randomized Search***

# ***Ensemble Methods***

# ***Analyze the Best Models and Their Errors***

# In[108]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[109]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# ***Evaluate Your System on the Test Set***

# In[110]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[111]:


final_rmse


# ***Launch, Monitor, and Maintain Your System***
