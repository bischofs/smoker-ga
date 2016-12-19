import random
import ipdb
import numpy
import numpy.matlib
from random import randint

from deap import base
from deap import creator
from deap import tools
numpy.set_printoptions(linewidth=400,suppress=None,threshold=10000)

IND_SIZE = 5

proj_types = {
    
    """1"""   'hd_bm': ['ebench', 'transient_tc'],
    """2"""    'ld_bm': ['steady_tc', 'humidity_ctrl', 'dac_sys ', 'ec_dyno ', 'oil_con ', 'cool_con ', 'air_con'],
    """3"""    '1065_cert': ['transient_tc', 'bg3', '1065_equip', 'dac_sys', 'ac_dyno', 'air_con'],
    """4"""    'fuels_lube_dev': ['transient_tc', 'dac_sys', 'ac_dyno', 'oil_con', 'cool_con', 'air_con'],
    """5"""    'frict_measure': ['transient_tc', 'dac_sys', 'ac_dyno', 'oil_con', 'cool_con', 'air_con'],
    """6"""    'engine_dev_gas': ['transient_tc', 'dac_sys', 'ac_dyno', 'oil_con', 'cool_con', 'air_con', 'humidity_ctrl'],
    """7"""    'engine_dev_diesel': ['transient_tc', 'dac_sys', 'ac_dyno', 'oil_con', 'cool_con', 'air_con', 'humidity_ctrl'],
    """8"""    'durability_hd': ['steady_tc', 'dac_sys', 'ec_dyno'],
    """9"""    'durability_ld': ['steady_tc', 'dac_sys', 'ec_dyno'],
    """10"""    'high_durability_hd': ['transient_tc', 'dac_sys', 'ac_dyno'],
    """11"""    'high_durability_ld': ['transient_tc', 'dac_sys', 'ac_dyno'],
    """12"""    'nvh': ['nvh_tc', 'dac_sys', 'ac_dyno'],
    """13"""    'marine_noncert': ['transient_tc', 'dac_sys', 'ac_dyno'],
    """14"""    'hybrid_dev': ['transient_tc', 'dac_sys', 'ac_dyno', 'orion'],
    """15"""    'after_treat_dev': ['transient_tc', 'dac_sys', 'ac_dyno', 'orion']
}


proj_index = {
    1:'hd_bm',
    2:'ld_bm',
    3:'1065_cert',
    4:'fuels_lube_dev',
    5:'frict_measure',
    6:'engine_dev_gas',
    7:'engine_dev_diesel',
    8:'durability_hd',
    9:'durability_ld',
    10:'high_durability_hd',
    11:'high_durability_ld',
    12:'nvh',
    13:'marine_noncert',
    14:'hybrid_dev',
    15:'after_treat_dev'
}





assets = {
    'ebench': 15,
    'transient_tc': 8,
    'steady_tc':4,
    'nvh_tc': 1,
    'ftir': 7,
    'bg3': 5,
    '1065_equip': 5,
    'humidity_ctrl': 4,
    'dac_sys': 14,
    'ac_dyno': 10,
    'ec_dyno': 35,
    'oil_con': 7,
    'cool_con': 8,
    'air_con': 4,
    'orion': 3
}

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


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax )

def initMatrix(ind_class):

    mtrx = numpy.matlib.zeros([26, 52])
    ind = ind_class(mtrx.A)

    for proj in initialProjects:
        length = proj[2][1] - proj[1][1]
        while(length > 0):
            ind[proj[1][0], (proj[2][1] - length)] = proj[0]
            length -= 1
            
    for i in range(26):

        proj_length = randint(20,230) / 7
        proj_type = randint(1,15)
        counter = -1

        for j in range(52):
            if(ind[i,j] == 0):
               counter += 1
            if(counter == proj_length):
                while(proj_length > 0):
                    ind[i,j - proj_length] = proj_type
                    proj_length -= 1
                            
    return ind 


def fitness_evaluate(individual):

    proj_list = []

    for i in range(26):
        for j in range(52):
    
            if proj_list and individual[i][j] not in proj_list: 
                proj_list.append(individual[i][j])
            elif not proj_list:
                proj_list.append(individual[i][j])
                

    for proj in proj_list:

        import ipdb; ipdb.set_trace()
        
        project_type = proj_index[int(proj - 1)]  
        
            

                
    fitness = 0

    



    
    return (fitness, )


toolbox = base.Toolbox()

toolbox.register("individual", initMatrix, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_evaluate)

pop = toolbox.population(n=300)

NGEN = 100

for g in range(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = map(toolbox.clone, offspring)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring





