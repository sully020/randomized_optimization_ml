from mlrose import random_hill_climb, simulated_annealing, genetic_alg, NeuralNetwork, GeomDecay
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import pandas as pd


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
    choice = input("Please enter the number of the algorithm you would like to use:\n 1.) R.R. Hill Climbing 2.) Simulated Annealing\
     3.) Genetic Algorithm \n")
    
    match choice:
        case '1':
            hill_climb_weights()
        case '2':
            simulate_annealing_weights()
        case '3':
            genetic_alg_weights()


def hill_climb_weights():
    neural_net = NeuralNetwork([3], algorithm = random_hill_climb, clip_max = 1, restarts = 100, curve = True)

def simulate_annealing_weights():
    neural_net = NeuralNetwork([3], algorithm = simulated_annealing, clip_max = 1, schedule = GeomDecay(), curve = True)

def genetic_alg_weights():
    neural_net = NeuralNetwork([3], algorithm = genetic_alg, clip_max = 1, pop_size = 100, curve = True)
