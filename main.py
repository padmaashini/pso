import numpy as np
import random
import copy
from scipy.optimize import dual_annealing, minimize


# Cost Functions
def rastrigin_function(x):
    A = 10
    n = len(x)

    to_sum = [(xi**2 - A*np.cos(2*np.pi*xi)) for xi in x]
    return A*n + sum(to_sum)

def sphere_function(x):
    to_sum = [xi**2 for xi in x]
    return sum(to_sum)

# PSO Algorithm
def pso(cost_function, initial_particle_positions, initial_velocities, max_iter):
    num_particles = len(initial_particle_positions)
    dimension = len(initial_particle_positions[0])

    # since we're trying to minimize the cost, the inital values will be set to inf
    individual_best = [float('inf')] * num_particles
    individual_best_positions = [copy.deepcopy(initial_particle_positions[i]) for i in range(num_particles)]

    particle_positions = copy.deepcopy(initial_particle_positions)
    particle_velocities = copy.deepcopy(initial_velocities)

    global_best = float('inf')
    global_best_position = None
    
    for _ in range(max_iter):
        for index, particle_position in enumerate(initial_particle_positions): 
            # calculate the fitness values of each particle
            fitness_value = cost_function(particle_position)

            # update individual values
            if individual_best[index] > fitness_value: 
                individual_best[index] = fitness_value
                individual_best_positions[index] = particle_position

            # update global
            if global_best > fitness_value: 
                global_best = fitness_value
                global_best_position = particle_position
            
            # calculate updated particle velocities
            # note that below, the ^ doesn't denote power, but rather convention of displaying
            # v(i)^(k+1) := alpha*v(i)^k + alpha2 * [(x(localBest)^k) - x(i)^k] + alpha3 * [(x(globalBest)^k) - x(i)^k]
            # alphas are a random value between 0 - 1 --> use random.random() to get this value
            particle_velocities[index] = random.random() * particle_velocities[index] + \
                                            random.random() * (individual_best_positions[index] - particle_position) + \
                                            random.random() * (global_best_position - particle_position)
            # calculate updated particle positions
             # note that below, the ^ doesn't denote power, but rather convention of displaying
            # x(i)^(k+1) := x(i)^k + v(i)^(k+1)
            particle_positions[index] = particle_position + particle_velocities[index]

    return global_best_position, global_best
            

# Run functions to test
if __name__ == "__main__":

    # Define the number of iterations
    num_iterations = 100

    rastrigin_fn_output = rastrigin_function([0.5, -1.5])
    sphere_output = sphere_function([0.5, -1.5])

    num_particles = 30
    dimension = 2 

    # Arrays to store results for each optimization method
    results_rastrigin_da = []
    results_sphere_da = []
    results_rastrigin_slsqp = []
    results_sphere_slsqp = []
    results_rastrigin_pso = []
    results_sphere_pso = []

    # Iterating through the optimization process
    for _ in range(num_iterations):
        # Initialize positions for Rastrigin Function within [-5.12, 5.12]
        initial_positions_rastrigin = np.random.uniform(-5.12, 5.12, (num_particles, dimension))

        # Initialize positions for Sphere Function within [-10, 10]
        initial_positions_sphere = np.random.uniform(-10, 10, (num_particles, dimension))

        velocity_range = 0.1 
        initial_velocities = np.random.uniform(-velocity_range, velocity_range, (num_particles, dimension))

        global_best_position, global_best_value = pso(rastrigin_function, initial_positions_rastrigin, initial_velocities, 100)
        # print("Best position for Rastrigin:", global_best_position)
        # print("Best value for Rastrigin:", global_best_value)
        results_rastrigin_pso.append(global_best_value)

        global_best_position, global_best_value = pso(sphere_function, initial_positions_sphere, initial_velocities, 100)
        # print("Best position for Sphere:", global_best_position)
        # print("Best value for Sphere:", global_best_value)
        results_sphere_pso.append(global_best_value)

        # Define the bounds for the Rastrigin Function
        bounds = [(-5.12, 5.12)] * 2

        # Use the simulated_annealing function
        result = dual_annealing(rastrigin_function, maxiter = 100, bounds=bounds)
        results_rastrigin_da.append(result.fun)

        # Define the bounds for the Sphere Function
        bounds = [(-10, 10)] * 2
        result = dual_annealing(sphere_function, maxiter = 100, bounds=bounds)
        results_sphere_da.append(result.fun)

        # Call the minimize function with SLSQP method
        initial_guess = np.random.uniform(-5.12, 5.12, dimension)
        result = minimize(rastrigin_function, initial_guess, method='SLSQP', options={'maxiter': 100})
        results_rastrigin_slsqp.append(result.fun)

        initial_guess = np.random.uniform(-10, 10, dimension)
        result = minimize(sphere_function, initial_guess, method='SLSQP', options={'maxiter': 100})
        results_sphere_slsqp.append(result.fun)

    # Calculate the average for each optimization method
    average_rastrigin_pso = np.mean(results_rastrigin_pso)
    average_sphere_pso = np.mean(results_sphere_pso)
    average_rastrigin_da = np.mean(results_rastrigin_da)
    average_sphere_da = np.mean(results_sphere_da)
    average_rastrigin_slsqp = np.mean(results_rastrigin_slsqp)
    average_sphere_slsqp = np.mean(results_sphere_slsqp)

    # Print the results
    print("\nAverage Optimal Value for Rastrigin (PSO):", average_rastrigin_pso)
    print("Average Optimal Value for Sphere (PSO):", average_sphere_pso)
    print("Average Optimal Value for Rastrigin (DA):", average_rastrigin_da)
    print("Average Optimal Value for Sphere (DA):", average_sphere_da)
    print("Average Optimal Value for Rastrigin (SLSQP):", average_rastrigin_slsqp)
    print("Average Optimal Value for Sphere (SLSQP):", average_sphere_slsqp)
