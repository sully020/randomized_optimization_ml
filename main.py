from mlrose import NeuralNetwork, random_hill_climb, simulated_annealing, genetic_alg
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import pandas as pd



def preprocess_data():
    patients = pd.read_csv('diabetes.csv')
    x = patients.drop(columns = 'class_val')
    y = patients['class_val']
    scaler = MaxAbsScaler()
    scaled_x = scaler.fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(scaled_x, y)
    ml_data = (xtrain, xtest, ytrain, ytest)
    return ml_data


def choose_optimizer():
    choice = input("Please enter the number of the algorithm you would like to use:\n 1.) R.R. Hill Climbing 2.) Simulated Annealing\
     3.) Genetic Algorithm \n")

def optimize_weights():
    optimizer = choose_optimizer()
    neural_net = NeuralNetwork(algorithm = optimizer, clip_max = 1, restarts = 100, schedule = 1, pop_size = 500, curve = True)
