""" 
Randomized Optimization Analysis - main.py
Christopher D. Sullivan
Professor Brian O'Neill
Due 4/1/23

This script is designed to model the differences of three randomized optimization algorithms
over three optimization problems each. This is achieved through mlrose, a python package 
designed for machine learning weight optimization, which contains other 
general fitness functions used for maximization/minimization.

Imports:
https://mlrose.readthedocs.io/en/stable/ - mlrose
https://scikit-learn.org/stable/ - scikit learn
https://pandas.pydata.org/ - pandas
https://matplotlib.org/    - matplotlib
https://docs.python.org/3/library/timeit.html - timeit

Data:
https://www.kaggle.com/code/mathchi/diagnostic-a-patient-has-diabetes/notebook : Mehmet Akturk
   - Classifies 700+ patients on whether or not they test + for diabetes. 
"""

import mlrose as ml
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import timeit

ml_imp = "import mlrose as ml" # Global statement variable used across timing functions.


# Scale, seperate, and package dataset.
def preprocess_data():
    patients = pd.read_csv('diabetes.csv')
    x = patients.drop(columns = 'class_val')
    y = patients['class_val']
    scaler = MaxAbsScaler()      # Method of scaling sparse data.
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)
    scaled_train = scaler.fit_transform(xtrain) # Avoids bias by only fitting to training data.
    scaled_test = scaler.transform(xtest)
    ml_data = (scaled_train, scaled_test, ytrain, ytest) # Return tuple(4) containing partitioned dataset.
    return ml_data

# Take user input to determine which weight optmization algorithm is used.
def choose_weights_optimizer():
    choice = input("Please enter the number of the algorithm you would like to use:\n1.) R.R. Hill Climbing 2.) Simulated Annealing\
 3.) Genetic Algorithm \n")

    match choice:
        case '1':
            print("You selected: Random Restart Hill Climbing")
            neural = ml.NeuralNetwork([67], algorithm = 'random_hill_climb', clip_max = 1, restarts = 200,
                max_iters = 1000, random_state = 0, curve = True)
        case '2':
            print("You selected: Simulated Annealing")
            neural = ml.NeuralNetwork([67], algorithm = 'simulated_annealing', clip_max = 1, schedule = ml.GeomDecay(), 
                max_iters = 1000, random_state = 0, curve = True)
        case '3':
            print("You selected: Genetic Algorithm")
            neural = ml.NeuralNetwork([67], algorithm = 'genetic_alg', clip_max = 1, pop_size = 200, 
                mutation_prob = 0.1, max_iters = 1000, random_state = 0, curve = True)
    return neural

# Uses selcted algorithm to search for optimal NN weight vector.
def optimize_weights(data):
    neural = choose_weights_optimizer()
    neural.fit(data[0], data[2])       # Fit weights to training data using chosen optimization alg.
    y_train_preds = neural.predict(data[0])
    train_accuracy = f1_score(data[2], y_train_preds)
    print("The F1-score training accuracy of the neural network was: " + "{:.2f}".format(train_accuracy) + "%")
    y_test_preds = neural.predict(data[1])
    test_accuracy = f1_score(data[3], y_test_preds)
    print("The F1-score test accuracy of the neural network was: " + "{:.2f}".format(test_accuracy) + "%")
    plt.plot(neural.fitness_curve)
    plt.ylabel('Relative Fitness Found')  # Create/display fitness curve plot for selected algorithm.
    plt.show()


# Unique optimization problems, each of n = 50

# Four peaks bitstring problem applied, display results
def test_four_peaks():
    problem = ml.FourPeaks(t_pct = 0.2) # Tail/head sequence of 10+
    best_hc_state = ml.random_hill_climb(ml.DiscreteOpt(50, problem), max_iters = 500, max_attempts = 500, restarts = 250)
    best_sa_state = ml.simulated_annealing(ml.DiscreteOpt(50, problem), max_iters = 500, max_attempts = 500, schedule = ml.GeomDecay())
    best_ga_state = ml.genetic_alg(ml.DiscreteOpt(50, problem), max_iters = 500, max_attempts = 500, pop_size = 250)
    print("Max Four Peaks fitness found using Random Hill Climb was: " + str(best_hc_state[1]))
    print("Max Four Peaks fitness found using Simulated Annealing was: " + str(best_sa_state[1]))
    print("Max Four Peaks fitness found using the Genetic Algorithm was: " + str(best_ga_state[1]))

# n-Queens chess problem applied, display results
def test_n_queens():
    problem = ml.Queens()
    best_hc_state = ml.random_hill_climb(ml.DiscreteOpt(50, problem), max_iters = 750, max_attempts = 500, restarts = 250)
    best_sa_state = ml.simulated_annealing(ml.DiscreteOpt(50, problem), max_iters = 750, max_attempts = 500, schedule = ml.GeomDecay())
    best_ga_state = ml.genetic_alg(ml.DiscreteOpt(50, problem), max_iters = 750, max_attempts = 500, pop_size = 250)
    print("Min. collision pairs found using Random Hill Climb was: " + str(best_hc_state[1]))
    print("Min. collision pairs found using Simulated Annealing was: " + str(best_sa_state[1]))
    print("Min. collision pairs found using the Genetic Algorithm was: " + str(best_ga_state[1]))

""" 
The following two functions find the lower bound runtime of each algorithm
by repeatedly running and testing them on the given problem. They return the lowest
value found across many executions, as variance is system-based and thus a lower bound is a
more accurate representation for comparisons. (timeit docs)
"""
def time_four_peaks():
    print("Four Peaks:")
    hc_test = "ml.random_hill_climb(ml.DiscreteOpt(50, ml.FourPeaks()), max_iters = 500, max_attempts = 500, restarts = 250)"
    sa_test = "ml.simulated_annealing(ml.DiscreteOpt(50, ml.FourPeaks()), max_iters = 500, max_attempts = 500)"
    ga_test = "ml.genetic_alg(ml.DiscreteOpt(50, ml.FourPeaks()), max_iters = 500, max_attempts = 500, pop_size = 250)"
    print("Random Hill Climb's lower bound runtime: " +
          str("{:.3f}".format(min(timeit.repeat(stmt = hc_test, setup = ml_imp, number = 5, repeat = 5)))) + " seconds.")
    print("Simulated Annealing's lower bound runtime: " +
          str("{:.3f}".format(min(timeit.repeat(stmt = sa_test, setup = ml_imp, number = 5, repeat = 5)))) + " seconds.")
    print("Genetic Algorithm's lower bound runtime: " + 
          str("{:.3f}".format(min(timeit.repeat(stmt = ga_test, setup = ml_imp, number = 5, repeat = 5)))) + " seconds.")
    print("----------------------------------")


def time_n_queens():
    hc_test = "ml.random_hill_climb(ml.DiscreteOpt(50, ml.Queens()), max_iters = 750, max_attempts = 500, restarts = 250)"
    sa_test = "ml.simulated_annealing(ml.DiscreteOpt(50, ml.Queens()), max_iters = 750, max_attempts = 500)"
    ga_test = "ml.genetic_alg(ml.DiscreteOpt(50, ml.Queens()), max_iters = 750, max_attempts = 500, pop_size = 250)"
    print("N Queens: ")
    print("Random Hill Climb's lower bound runtime: " 
          + str("{:.3f}".format(min(timeit.repeat(stmt = hc_test, setup = ml_imp, number = 5, repeat = 5))))  
          + " seconds.")
    print("Simulated Annealing's lower bound runtime: " 
          + str("{:.3f}".format(min(timeit.repeat(stmt = sa_test, setup = ml_imp, number = 5, repeat = 5))))
          + " seconds.")
    print("Genetic Algorithm's lower bound runtime: " 
          + str("{:.3f}".format(min(timeit.repeat(stmt = ga_test, setup = ml_imp, number = 5, repeat = 5))))
          + " seconds.")
    print("----------------------------------")


# Output results of all problems.
def main():
    data = preprocess_data()
    optimize_weights(data)
    test_four_peaks()
    test_n_queens()
    time_four_peaks()
    time_n_queens()

if __name__ == "__main__":
    main()