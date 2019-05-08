import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn


def prepare_country_stats(oecd, gdp):
    merge = pd.merge(gdp, oecd, how='outer', on='Country')
    return merge


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
# Visualize the data
country_stats.plot(kind='scatter', x='Unnamed: 7', y='Value')
plt.show()
# Select a linear model
lin_reg_model = sklearn.linear_model.LinearRegression()
# Train the model
lin_reg_model.fit(X, y)
# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(lin_reg_model.predict(X_new))  # outputs [[ 5.96242338]]
