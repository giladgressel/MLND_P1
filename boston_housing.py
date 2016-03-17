"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

################################
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
################################


def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    ###################################

    number_of_houses = np.size(housing_prices)
    number_of_features = np.size(housing_features[0])
    min_housing_price = np.min(housing_prices)
    max_housing_price = np.max(housing_prices)
    mean_housing_price = np.mean(housing_prices)
    median_housing_price = np.median(housing_prices)
    std_housing_prices = np.std(housing_prices)

    print " ground truth is :"
    print "the number of houses in the data is : ", number_of_houses
    print "the number of features for the data is : ", number_of_features
    print "the minimum price of a house is : ", min_housing_price
    print "the maximum price of a house is : ", max_housing_price
    print "the mean of the prices is : ", mean_housing_price
    print "the median of the price is : ", median_housing_price
    print "the standard deviation of a house-price is : ", std_housing_prices


    ###################################

    # Please calculate the following values using the Numpy library
    # Size of data (number of houses)?
    # Number of features?
    # Minimum price?
    # Maximum price?
    # Calculate mean price?
    # Calculate median price?
    # Calculate standard deviation?


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    ###################################
    return mean_squared_error(label, prediction)
    ###################################

    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    pass


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    ###################################
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.30, random_state= 1)

    # comparing predictions to ground truth here.
    print ""
    print ""
    print "Predictions with a 70/30 split are : "
    print ""

    clf = DecisionTreeRegressor(max_depth=4)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    min_prediction_price = np.min(predictions)
    max_prediction_price = np.max(predictions)
    mean_prediction_price = np.mean(predictions)
    median_prediction_price = np.median(predictions)
    std_prediction_prices = np.std(predictions)

    print "the minimum price of a house is : ", min_prediction_price
    print "the maximum price of a house is : ", max_prediction_price
    print "the mean of the prices is : ", mean_prediction_price
    print "the median of the price is : ", median_prediction_price
    print "the standard deviation of a house-price is : ", std_prediction_prices
    print ""
    print ""


    ###################################

    return X_train, y_train, X_test, y_test


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.linspace(1, len(X_train), 50)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:int(s)], y_train[:int(s)])

        # Find the performance on the training and testing set
        train_err[int(i)] = performance_metric(y_train[:int(s)], regressor.predict(X_train[:int(s)]))
        test_err[int(i)] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err, depth)



def learning_curve_graph(sizes, train_err, test_err, depth):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size, Max_Depth is : %s' %depth)
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()
    return pl


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()


def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {'max_depth':(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}

    ###################################

    mse = make_scorer(mean_squared_error, greater_is_better=False)
    clf = GridSearchCV(regressor, parameters, scoring=mse)

    ###################################

    # 1. Find the best performance metric
    # should be the same as your performance_metric procedure
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html

    # 2. Use gridearch to fine tune the Decision Tree Regressor and find the best model
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

    # Fit the learner to the training data
    print "Final Model: "
    print clf.fit(X, y)

    print ""
    print "best estimator"
    print clf.best_estimator_
    print ""


    # compare gridsearchCV prediction to the ground-truth.
    predictions = clf.predict(X)

    min_prediction_price = np.min(predictions)
    max_prediction_price = np.max(predictions)
    mean_prediction_price = np.mean(predictions)
    median_prediction_price = np.median(predictions)
    std_prediction_prices = np.std(predictions)

    print ""
    print "predictions with training on the full data set"
    print ""
    print "the minimum price of a house is : ", min_prediction_price
    print "the maximum price of a house is : ", max_prediction_price
    print "the mean of the prices is : ", mean_prediction_price
    print "the median of the price is : ", median_prediction_price
    print "the standard deviation of a house-price is : ", std_prediction_prices


    # Use the model to predict the output of a particular sample

    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = clf.predict(x)
    print ""
    print "House: " + str(x)
    print "Prediction: " + str(y)


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    explore_city_data(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs
    max_depths = [1,2,3,4,5,6,7,8,9,10]
    for max_depth in max_depths:
          learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)


if __name__ == "__main__":
    main()
