import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model as sklm


def prepare_country_stats(oecd, gdp):
    merge = pd.merge(gdp, oecd, how='outer', on='Country')
    return merge


def clear_NaN_values(Xc, yc):
    Xc = Xc[np.logical_not(np.isnan(yc))]
    yc = yc[np.logical_not(np.isnan(yc))]
    yc = yc[np.logical_not(np.isnan(Xc))]
    Xc = Xc[np.logical_not(np.isnan(Xc))]
    return Xc.reshape((-1, 1)), yc.reshape((-1, 1))


# Load the data
oecd_bli = pd.read_csv("../Data/in/oecd_bli_2017_simplified.csv",
                       thousands=',')  # simplified version contain only Life satisfaction values
gdp_per_capita = pd.read_csv("../Data/in/gdp_per_capita.csv", thousands=',',
                             encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
# Printing the full table
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(country_stats)
X = np.c_[country_stats["Unnamed: 7"]]  # GDP
y = np.c_[country_stats["Value"]]  # Life satisfaction

X, y = clear_NaN_values(X, y)
# Visualize the data

country_stats.plot(kind='scatter', x='Unnamed: 7', y='Value')
plt.show()
# Select a linear model
lin_reg_model = sklm.LinearRegression()

# Train the model
lin_reg_model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(lin_reg_model.predict(X_new))  # outputs [[6.24626328]]

# Trying a KNN (average of the N nearest neighbor, here N = 3

clf = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
clf.fit(X, y)
print(clf.predict(X_new))  # outputs [[5.8]]

