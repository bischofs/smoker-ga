from scoop import futures
import multiprocessing

import random
import ipdb
import numpy
import numpy.matlib
from rdrand import RdRandom
import os
r = RdRandom()

from decimal import *
getcontext().prec = 30

from deap import base
from deap import creator
from deap import tools, algorithms 
numpy.set_printoptions(linewidth=400, suppress=None, threshold=10000)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax )

IND_SIZE = 5

proj_types = {

    'hd_bm': ['ebench', 'dac_sys'],
    'ld_bm': [ 'humidity_ctrl', 'dac_sys', 'oil_con', 'cool_con', 'air_con'],
    '1065_cert': ['bg3', 'ebench', '1065_equip', 'dac_sys', 'air_con'],
    'fuels_lube_dev': ['dac_sys', 'oil_con', 'cool_con', 'air_con'],
    'frict_measure': ['dac_sys', 'oil_con', 'cool_con', 'air_con'],
    'engine_dev_gas': ['dac_sys',  'oil_con', 'cool_con', 'air_con', 'humidity_ctrl'],
    'engine_dev_diesel': ['dac_sys', 'oil_con', 'cool_con', 'air_con', 'humidity_ctrl'],
    'durability_hd': ['dac_sys'],
    'durability_ld': [ 'dac_sys'],
    'high_durability_hd': [ 'dac_sys'],
    'high_durability_ld': [ 'dac_sys'],
    'nvh': ['dac_sys','nvh_equip'],
    'marine_noncert': ['dac_sys'],
    'hybrid_dev': ['dac_sys', 'orion', 'battery_em'],
    'after_treat_dev': ['dac_sys', 'orion']
}


# if cell has fixed asset prefer this

# if project requires asset and cell has no fixed assedt use asset

assets = {
    'ebench': 15,
    'nvh_equip': 2, 
    'ftir': 7,
    'bg3': 5,
    '1065_equip': 5,
    'humidity_ctrl': 4,
    'dac_sys': 28,
    'oil_con': 7,
    'cool_con': 8,
    'air_con': 6,
    'air_con_fixed': 5,
    'orion': 3,
    'nvh_equip': 1,
    'battery_em': 1
}

#assets cell does not have
cell_constraints = {

    '1': ['1065_equip','humidity_ctrl', 'air_con_fixed'],
    '2': ['ac_dyno', 'battery_em','1065_equip','humidity_ctrl', 'air_con_fixed'],
    '3n':['ac_dyno','battery_em','1065_equip','humidity_ctrl', 'air_con_fixed'],
    '3s':['ac_dyno','battery_em','1065_equip','humidity_ctrl', 'air_con_fixed'],
    '4n':['battery_em','1065_equip'],
    '4s':['battery_em','1065_equip'],
    '5n':['battery_em','1065_equip'],
    '5s':['battery_em','1065_equip'],
    '6n':['battery_em','ac_dyno','1065_equip','humidity_ctrl', 'air_con_fixed'],
    '6s':['battery_em','ac_dyno','1065_equip','humidity_ctrl', 'air_con_fixed'],
    '7n':['battery_em'],
    '7s':['battery_em'],
    '8n':['battery_em','humidity_ctrl', 'air_con_fixed'],
    '8s':['battery_em','humidity_ctrl', 'air_con_fixed'],
    '9n':['battery_em','1065_equip'],
    '9s':['battery_em','1065_equip'],
    '10n':['battery_em'],
    '10s':['battery_em'],
    '11n':['battery_em','1065_equip','humidity_ctrl', 'air_con_fixed'],
    '11s':['battery_em','1065_equip','humidity_ctrl', 'air_con_fixed'],
    '12n':['battery_em','humidity_ctrl', 'air_con_fixed'],
    '12s':['battery_em','humidity_ctrl', 'air_con_fixed'],
    '13n':['humidity_ctrl', 'air_con_fixed'],
    '13s':['humidity_ctrl', 'air_con_fixed'],
    '14n':['battery_em','humidity_ctrl', 'air_con_fixed'],
    '14s':['battery_em','humidity_ctrl', 'air_con_fixed']
}

cell_index = [

    '1',
    '2',
    '3n',
    '3s',
    '4n',
    '4s',
    '5n',
    '5s',
    '6n',
    '6s',
    '7n',
    '7s',
    '8n',
    '8s',
    '9n',
    '9s',
    '10n',
    '10s',
    '11n',
    '11s',
    '12n',
    '12s',
    '13n',
    '13s',
    '14n',
    '14s'
]

proj_index = [
    'hd_bm',
    'ld_bm',
    '1065_cert',
    'fuels_lube_dev',
    'frict_measure',
    'engine_dev_gas',
    'engine_dev_diesel',
    'durability_hd',
    'durability_ld',
    'high_durability_hd',
    'high_durability_ld',
    'nvh',
    'marine_noncert',
    'hybrid_dev',
    'after_treat_dev'
]

initialProjects = [
    #1
    [12, [0,0], [0,15]],
#2
    [6, [1,0], [2,14]],
#3
    [1, [2,0], [2,51]],
    [1, [3,0], [3,51]],
#4
    [2, [4,0], [2,4]],
    [2, [5,0], [5,11]],
#5
    [4, [6,0], [6,8]],
    [5, [7,0], [7,8]],
#6
    [4, [8,0], [8,3]],
    [4, [9,0], [9,3]],
#7
    [3, [10,0], [10,12]],
    [2, [11,0], [11,1]],
#8
    [3, [12,0], [12,10]],
#9
    [3, [14,0], [14,8]],
    [3, [15,0], [15,16]],
#10
    [6, [16,0], [16,12]],
    [2, [17,0], [17,8]],
#11
    [1, [18,0], [18,52]],
    [1, [19,0], [19,52]],
#12
    [1, [20,0], [20,11]],
    [1, [21,0], [21,5]],
#13
    [6, [22,0], [22,17]],
#14
    [5, [24,0], [24,7]],
    [5, [25,0], [25,7]]
]


def initMatrix(ind_class):

    mtrx = numpy.matlib.zeros([26, 52])
    ind = ind_class(mtrx.A)

    #initial project 
    for proj in initialProjects:
        length = proj[2][1] - proj[1][1]
        while(length > 0):
            ind[proj[1][0], (proj[2][1] - length)] = proj[0]
            length -= 1

    #Stochastic fill 
    for i in range(26):
        for k in range(50):
            valid = False
            while(not valid):
                proj_length = r.randint(20,230) / 7
                proj_type = r.randint(1,15)
                counter = -1

                valid = check_valid(i, proj_type)

                if valid == True:
                    for j in range(52):
                        if(ind[i,j] == 0):
                            counter += 1
                        if(counter == proj_length):
                            while(proj_length > 0):
                                ind[i,j - proj_length] = proj_type
                                proj_length -= 1
                                
    return ind 

def check_valid(row, proj_type):

    invalid_assets = cell_constraints[cell_index[row]]
    needed_assets = proj_types[proj_index[proj_type - 1]]
    
    for needed_asset in needed_assets:
        for invalid_asset in invalid_assets:
            if needed_asset == invalid_asset:
                return False 

    return True 

def fitness_evaluate(individual):

    proj_list = []
    assets_copy = assets.copy()

    column_asset_sum = 0
    columns_asset_list = []
    
    for j in range(52):

        for i in range(26):

            project_type = proj_index[int(individual[i][j] - 1)]  
            used_assets = proj_types[project_type]

            for asset in used_assets:
                if(asset + '_fixed' not in cell_constraints[cell_index[i]]):   #cell has no fixed asset of type "asset"
                    assets_copy[asset] -= 1
                else:
                    assets_copy[asset + '_fixed'] -= 1  
            
        column_asset_sum = 0
        for item in assets_copy.items():
            column_asset_sum += abs(item[1])
            
        assets_copy = assets.copy()
        columns_asset_list.append(column_asset_sum)

        total_asset_sum = 0

    for column in columns_asset_list:
        total_asset_sum += column
        
    fitness = 1 / float(total_asset_sum)
    fitness *= 100

    individual.cols = columns_asset_list
    
    return (fitness,)


def mutate_rows(ind):

    row = r.randint(0,25)
    rand_proj = r.randint(1,15) 
    count = 0
    end_column = 0
    
    for i in range(51):

        if(ind[row, i] != ind[row, i + 1] and count == 0):
            start_column = i + 1
            count = 1
        elif(ind[row, i] != ind[row, i + 1] and count == 1):
            end_column = i + 1
            break

    if(end_column != 0):
        
        length = end_column - start_column
        
        for i in range(length):
            ind[row, start_column + i] = rand_proj

    return ind, 



toolbox = base.Toolbox()

toolbox.register("individual", initMatrix, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("evaluate", fitness_evaluate)
toolbox.register("mutate", mutate_rows)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=50)


def main():
    

    NGEN = 5000

    for g in range(NGEN):
    

        # Select and clone the next generation individuals
        offspring = toolbox.map(toolbox.clone, toolbox.select(pop, len(pop)))
    
        # Apply crossover and mutation on the offspring
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.5, mutpb=0.1)
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop[:] = offspring
    

    import ipdb;ipdb.set_trace()
    return pop



# if __name__ == '__main__':

#     toolbox.register('map', futures.map)
    
pop = main()


