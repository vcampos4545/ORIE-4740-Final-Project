import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

def clean_dataset(df):
    #Drop all categorical/useless columns
    numeric = df.drop(columns=['Unnamed: 0', 'Id', 'Name', 'StyleName', 'Region', 'Winery'])
    #Drop all records with body or acidity = -1
    numeric = numeric.loc[~((numeric["Body"] == -1) | (numeric["Acidity"] == -1) | (numeric["Year"] == 0))]
    numeric.head()

    #Making dummy columns for each country
    countries = numeric.Country.unique()

    for country in countries:
        numeric[country] = np.where(numeric["Country"] == country, 1, 0)

    #Remove country column and make final cleaned data
    data = numeric.drop(columns=["Country"])
    data.reset_index(inplace=True,drop=True)
    return data

def get_dataset(scale=False, pca=False, n_pc=15):
    df = pd.read_csv("WineData.csv")
    df_cleaned = clean_dataset(df)

    #Create X and y
    #Create log price column
    df_cleaned['log_price'] = np.log(df_cleaned["Price"])

    #X is every column but log price, y is only log price
    X = df_cleaned.drop(columns=["log_price","Price"])
    y = df_cleaned[["log_price"]].values.ravel()

    if scale:
        #Scale the data
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    if pca:
        #PCA transform the data
        pca_scaled = PCA(n_components=len(list(X.columns)))
        X_transformed = pca_scaled.fit_transform(X)
        X = X_transformed[:, :n_pc]

    return X, y

# get a list of models to evaluate
def get_models():
    models = list()
    models.append(LinearRegression())
    models.append(Ridge())
    models.append(Lasso())
    models.append(DecisionTreeRegressor())
    models.append(BaggingRegressor())
    models.append(RandomForestRegressor())
    models.append(GradientBoostingRegressor())
    return models

# evaluate the model using a given test condition
def evaluate_model(cv, model):
    # get the dataset
    #X, y = get_dataset(scale=True,pca=True) #Which dataset? scale, pca params
    # evaluate the model
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    # return scores
    return np.mean(scores)

X,y = get_dataset(scale=True, pca = True)

# define test conditions
#ideal_cv = LeaveOneOut()
ideal_cv = KFold(n_splits=100, shuffle=True, random_state=1)
cv = KFold(n_splits=10, shuffle=True, random_state=1)
# get the list of models to consider
models = get_models()
# collect results
ideal_results, cv_results = list(), list()
# evaluate each model
for model in models:
    # evaluate model using each test condition
    cv_mean = evaluate_model(cv, model)
    ideal_mean = evaluate_model(ideal_cv, model)
    # check for invalid results
    if np.isnan(cv_mean) or np.isnan(ideal_mean):
        continue
    # store results
    cv_results.append(cv_mean)
    ideal_results.append(ideal_mean)
    # summarize progress
    print('>%s: ideal=%.3f, cv=%.3f' % (type(model).__name__, ideal_mean, cv_mean))
# calculate the correlation between each test condition
corr, _ = pearsonr(cv_results, ideal_results)
print('Correlation: %.3f' % corr)
# scatter plot of results
plt.scatter(cv_results, ideal_results)
# plot the line of best fit
coeff, bias = np.polyfit(cv_results, ideal_results, 1)
line = coeff * np.asarray(cv_results) + bias
plt.plot(cv_results, line, color='r')
# label the plot
plt.title('10-fold CV vs LOOCV Mean Accuracy')
plt.xlabel('Mean Accuracy (10-fold CV)')
plt.ylabel('Mean Accuracy (LOOCV)')
# show the plot
plt.show()
