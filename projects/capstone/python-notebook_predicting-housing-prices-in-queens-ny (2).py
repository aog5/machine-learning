# Python Notebook - Predicting Housing Prices in Queens, NY

#Import standard Python libraries for numerical analysis and visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

#Set plot styles
sns.set(style="darkgrid")

housing = datasets["Queens Prices"]
housing.head()

housing.describe()

housing.info()

housing.drop(["borough_code", "maximum_allowable_residential_far", "floor_area_total_building", "floor_area_residential"], axis=1, inplace=True)

air_quality = { 11101 : 5.3, 11102 : 5.3, 11103 : 5.3, 11104 : 5.3, 11105 : 5.3, 11106 : 5.3, 11109 : 5.3, 11368 : 4.4, 11369 : 4.4, 11370 : 4.4, 11372 : 4.4, 11373 : 4.4, 11377 : 4.4, 11378 : 4.4, 11354 : 3.1, 11355 : 3.1, 11356 : 3.1, 11357 : 3.1, 11358 : 3.1, 11359 : 3.1, 11360 : 3.1, 11361 : 2.8, 11362 : 2.8, 11363 : 2.8, 11364 : 2.8, 11365 : 2.9, 11366 : 2.9, 11367 : 2.9, 11414 : 3.3, 11415 : 3.3, 11416 : 3.3, 11417 : 3.3, 11418 : 3.3, 11419 : 3.3, 11420 : 3.3, 11421 : 3.3, 11374 : 3.7, 11375 : 3.7, 11379 : 3.7, 11385 : 3.7, 11412 : 3, 11423 : 3, 11430 : 3, 11432 : 3, 11433 : 3, 11434 : 3, 11435 : 3, 11436 : 3, 11001 : 2.6, 11004 : 2.6, 11005 : 2.6, 11040 : 2.6, 11411 : 2.6, 11413 : 2.6, 11422 : 2.6, 11426 : 2.6, 11427 : 2.6, 11428 : 2.6, 11429 : 2.6, 11691 : 2.6, 11692 : 2.6, 11693 : 2.6, 11694 : 2.6, 11697 : 2.6 }
qns_eic = { 11004 : 0.484444444444444, 11005 : 0.258741258741259, 11101 : 0.541666666666667, 11102 : 0.611650485436893, 11103 : 0.604770813844715, 11104 : 0.599348534201954, 11105 : 0.573373676248109, 11106 : 0.596781403665624, 11354 : 0.777186564682408, 11355 : 0.851420678768745, 11356 : 0.709443099273608, 11357 : 0.491902834008097, 11358 : 0.623804463336876, 11360 : 0.427, 11361 : 0.548293391430646, 11362 : 0.467889908256881, 11363 : 0.446745562130178, 11364 : 0.53078677309008, 11365 : 0.618277010947168, 11366 : 0.55783308931186, 11367 : 0.630539241857982, 11368 : 0.838716271463614, 11369 : 0.762092238470191, 11370 : 0.693233082706767, 11372 : 0.703829787234043, 11373 : 0.790552410696857, 11374 : 0.564057717533887, 11375 : 0.426894045289349, 11377 : 0.716816094467527, 11378 : 0.616086708499715, 11379 : 0.517806670435274, 11385 : 0.678884137650173, 11411 : 0.609620721554117, 11412 : 0.680104712041885, 11413 : 0.639963167587477, 11414 : 0.496978851963746, 11415 : 0.528616024973985, 11416 : 0.774856203779786, 11417 : 0.689069925322471, 11418 : 0.722693831352575, 11419 : 0.782758620689655, 11420 : 0.716092455298735, 11421 : 0.722332015810277, 11422 : 0.635757575757576, 11423 : 0.687294272547729, 11426 : 0.551130247578041, 11427 : 0.591452991452992, 11428 : 0.652917505030181, 11429 : 0.692191053828658, 11430 : 0.5625, 11432 : 0.723125884016973, 11433 : 0.767813267813268, 11434 : 0.687152563310686, 11435 : 0.709000762776507, 11436 : 0.716981132075472, 11501 : 0.455463728191001, 11507 : 0.394179894179894, 11509 : 0.37593984962406, 11510 : 0.482700892857143, 11514 : 0.463035019455253, 11516 : 0.442136498516321, 11518 : 0.432142857142857, 11520 : 0.651919866444073, 11530 : 0.337698139214335, 11542 : 0.57089552238806, 11545 : 0.389438943894389, 11548 : 0.462686567164179, 11550 : 0.746967071057192, 11552 : 0.510852713178295, 11553 : 0.720274914089347, 11554 : 0.462316641375822, 11557 : 0.402439024390244, 11558 : 0.523584905660377, 11559 : 0.458549222797928, 11560 : 0.443148688046647, 11561 : 0.428205128205128, 11563 : 0.441840277777778, 11565 : 0.378378378378378, 11566 : 0.387622149837134, 11568 : 0.362573099415205, 11570 : 0.402777777777778, 11572 : 0.429371499688861, 11575 : 0.732749178532311, 11576 : 0.332807570977918, 11577 : 0.387687188019967, 11579 : 0.40530303030303, 11580 : 0.523744911804613, 11581 : 0.454376163873371, 11590 : 0.593233674272227, 11596 : 0.394160583941606, 11598 : 0.376689189189189, 11691 : 0.757805530776093, 11692 : 0.742196531791908, 11693 : 0.636842105263158, 11694 : 0.496590909090909, 11697 : 0.357843137254902, 11697 : 0.357843137254902}

x = housing["zipcode"]
housing["earned_income"] = x.map(qns_eic)
x2 = housing["zipcode"]
housing["air_qual"] = x2.map(air_quality)

housing.info()

null_vals = housing[housing["earned_income"].isnull()]
null_vals["zipcode"].unique()

housing.dropna(axis=0, how="any", inplace=True)

housing.info()

#Importing our sales data
sales13 = datasets["sales13"]
sales14 = datasets["sales14"]
sales15 = datasets["sales15"]
sales16 = datasets["sales16"]

#Finding the average sale price per group of home
qns_13 = sales13.groupby("TYPE_OF_HOME").mean()
qns_14 = sales14.groupby("TYPE_OF_HOME").mean()
qns_15 = sales15.groupby("TYPE_OF_HOME").mean()
qns_16 = sales16.groupby("TYPE_OF_HOME").mean()

#Selecting only data referring to single family homes and combining to form a dataset
qns_16 = qns_16[0:1]
qns_15 = qns_15[0:1]
qns_14 = qns_14[0:1]
qns_13 = qns_13[0:1]
qns_hist = pd.concat([qns_13, qns_14, qns_15, qns_16], keys=['2013', '2014', '2015', '2016'])
qns_hist.head()

#Minimum sales price per year
min_13 = sales13["LOWEST_SALE_PRICE"].min()
min_14 = sales14["LOWEST_SALE_PRICE"].min()
min_15 = sales15["LOWEST_SALE_PRICE"].min()
min_16 = sales16["LOWEST_SALE_PRICE"].min()

#Maximim sales price per year 
max_13 = sales13["HIGHEST_SALE_PRICE"].max()
max_14 = sales14["HIGHEST_SALE_PRICE"].max()
max_15 = sales15["HIGHEST_SALE_PRICE"].max()
max_16 = sales16["HIGHEST_SALE_PRICE"].max()

print("The minimum sale prices for homes between 2013 & 2016 are as follows:\n")
print("2013: ${}".format(min_13))
print("2014: ${}".format(min_14))
print("2015: ${}".format(min_15))
print("2016: ${}".format(min_16))

print("The maximum sale prices for homes between 2013 & 2016 are as follows:\n")
print("2013: ${}".format(max_13))
print("2014: ${}".format(max_14))
print("2015: ${}".format(max_15))
print("2016: ${}".format(max_16))

plt.figure(figsize=(15, 5))
sns.distplot((housing["sale_price"]),bins=250);

min_price = np.min(housing["sale_price"])
avg_price = np.mean(housing["sale_price"])
median_price = np.median(housing["sale_price"])
max_price = np.max(housing["sale_price"])
print("Sales data for the year 2017-2018: \n")
print("Minimum price: ${}".format(min_price))
print("Mean price: ${}".format(avg_price))
print("Median price ${}".format(median_price))
print("Maximum price: ${}".format(max_price))

housing["neighborhood"] = housing["neighborhood"].astype('category')
housing["zipcode"] = housing["zipcode"].astype('category')
housing["school_district"] = housing["school_district"].astype('category')
housing["community_district"] = housing["community_district"].astype('category')

housing.info()

plt.figure(figsize=(15, 15))
sns.stripplot(x="x_coordinate", y="y_coordinate", hue="sale_price", data=housing)

housing.corr()["sale_price"].sort_values()

sns.regplot(x="earned_income", y="sale_price", data=housing)

sns.regplot(x="land_square_feet", y="sale_price", data=housing)

sns.regplot(x="gross_square_feet", y="sale_price", data=housing)

prices = housing["sale_price"]

features = housing.drop(["neighborhood", "zipcode", "sale_price", "community_district", "school_district", "air_qual"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=.2, random_state=147)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

pd.plotting.scatter_matrix(X_train, alpha = 0.3, figsize = (15,15), diagonal = 'kde');

import matplotlib.pyplot as pl
import numpy as np
import sklearn.learning_curve as curves
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import ShuffleSplit, train_test_split

def ModelLearning(X, y):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = pl.figure(figsize=(10,7))

    # Create three different models based on max_depth
    for k, depth in enumerate([1,3,5,10]):
        
        # Create a Decision tree regressor at max_depth = depth
        regressor = DecisionTreeRegressor(max_depth = depth)

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(regressor, X, y, \
            cv = cv, train_sizes = train_sizes, scoring = 'r2')
        
        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)

        # Subplot the learning curve 
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')
        
        # Labels
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])
    
    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()


def ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, test_scores = curves.validation_curve(DecisionTreeRegressor(), X, y, \
        param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'r2')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Regressor Complexity Performance')
    pl.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
    pl.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
    pl.fill_between(max_depth, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    pl.fill_between(max_depth, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')
    
    # Visual aesthetics
    pl.legend(loc = 'lower right')
    pl.xlabel('Maximum Depth')
    pl.ylabel('Score')
    pl.ylim([-0.05,1.05])
    pl.show()


def PredictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting data. """

    # Store the predicted prices
    prices = []

    for k in range(10):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size = 0.2, random_state = k)
        
        # Fit the data
        reg = fitter(X_train, y_train)
        
        # Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)
        
        # Result
        print "Trial {}: ${:,.2f}".format(k+1, pred)

    # Display price range
    print "\nRange in prices: ${:,.2f}".format(max(prices) - min(prices))


ModelLearning(features, prices)

ModelComplexity(X_train, y_train)

from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, ShuffleSplit

def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits=10, test_size=.1, train_size=None, random_state=None)
    regressor = DecisionTreeRegressor()
    params = {'max_depth': list(range(1, 11))}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    grid = grid.fit(X, y)
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

#Select supervised learning algorithms to compare
from sklearn.linear_model import LinearRegression

reg1 = LinearRegression(normalize=True)
reg1.fit(X_train, y_train)
reg1.score(X_train, y_train)

y_1 = reg1.predict(X_train)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

reg2 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=9), n_estimators=300, random_state=17)
reg2.fit(X_train, y_train.values.ravel())
reg2.score(X_train, y_train.values.ravel())

y_2 = reg2.predict(X_train)

from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.19.1: ShuffleSplit(n_splits=10, test_size=’default’, train_size=None, random_state=None)
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(n_splits=10, test_size=.1, train_size=None, random_state=None)
    
    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': list(range(1, 11))}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

lsf = features["land_square_feet"]
gsf = features["gross_square_feet"]
eic = features["earned_income"]

def client_features(f1, f2, f3, n):
  client_data = []
  f1 = np.
  f2 = 

f1 = np.random.random_integers(lsf.min(), lsf.max(), 3)
f2 = np.random.random_integers(gsf.min(), gsf.max(), 3)
f3 = np.random.random_integers(eic.min(), eic.max(), 3)

client_1 = f1[0] + f2[1] + f3[1]
client_2 = f1[2], f2[2], f3[0]
client_3 = f1[1], f2[0], f3[2]
client_data = client_1 + client_2 + client_3
print(client_data)

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

PredictTrials(features, prices, fit_model, client_data)

