import mlrose as ml
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt


def preprocess_data():
    patients = pd.read_csv('diabetes.csv')
    x = patients.drop(columns = 'class_val')
    y = patients['class_val']
    scaler = MaxAbsScaler()
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)
    scaled_train = scaler.fit_transform(xtrain) # Avoids bias by only fitting to training data.
    scaled_test = scaler.transform(xtest)
    ml_data = (scaled_train, scaled_test, ytrain, ytest)
    return ml_data


def choose_weights_optimizer():
    choice = input("Please enter the number of the algorithm you would like to use:\n1.) R.R. Hill Climbing 2.) Simulated Annealing\
 3.) Genetic Algorithm \n")
    
    # Random state?

    match choice:
        case '1':
            print("You selected: Random Restart Hill Climbing")
            restarts = int(input("How many random restarts would you like to use? "))
            neural = ml.NeuralNetwork([67], algorithm = 'random_hill_climb', clip_max = 1, restarts = restarts,
                max_iters = 500, random_state = 0, curve = True)
        case '2':
            print("You selected: Simulated Annealing")
            neural = ml.NeuralNetwork([67], algorithm = 'simulated_annealing', clip_max = 1, schedule = ml.GeomDecay(), 
                max_iters = 500, random_state = 0, curve = True)
        case '3':
            print("You selected: Genetic Algorithm")
            population = int(input("Please enter population size: "))
            neural = ml.NeuralNetwork([67], algorithm = 'genetic_alg', clip_max = 1, pop_size = population, 
                mutation_prob = 0.1, max_iters = 500, random_state = 0, curve = True)
    return neural


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
    plt.ylabel('Relative Fitness Found')
    plt.show()


def test_four_peaks():
    problem = ml.FourPeaks(t_pct = 0.2) # Tail/head sequence of 10+
    best_hc_state = ml.random_hill_climb(ml.DiscreteOpt(50, problem), max_iters = 5000, max_attempts = 500, restarts = 200)
    best_sa_state = ml.simulated_annealing(ml.DiscreteOpt(50, problem), max_iters = 5000, max_attempts = 500)
    best_ga_state = ml.genetic_alg(ml.DiscreteOpt(50, problem), max_iters = 5000, max_attempts = 500, pop_size = 500)
    print("Max Four Peaks fitness found using Random Hill Climb was: " + str(best_hc_state[1]))
    print("Max Four Peaks fitness found using Simulated Annealing was: " + str(best_sa_state[1]))
    print("Max Four Peaks fitness found using the Genetic Algorithm was: " + str(best_ga_state[1]))


def test_n_queens():
    pass

def main():
    data = preprocess_data()
    optimize_weights(data)

if __name__ == "__main__":
    main()