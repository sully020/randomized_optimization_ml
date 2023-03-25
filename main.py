import mlrose
import pandas

def choose_optimizer():
    choice = input("Please enter the number of the algorithm you would like to use:\n 1.) R.R. Hill Climbing 2.) Simulated Annealing\
     3.) Genetic Algorithm \n")

def optimize_weights():
    optimizer = choose_optimizer()
    neural_net = mlrose.NeuralNetwork(67, algorithm = optimizer, clip_max = 1, random_state = 1, restarts = 100, schedule = 1, pop_size = 500, curve = True)
