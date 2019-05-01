#!./bin/python
'''
Code By Michael Sherif Naguib
License: MIT
Date: 4/12/19
@University of Tulsa
Description:  (Multiprocessing on 1 PC)


Please read the description in main.py as it discusses the scope of the project and motivation.
            this code is responsible for distributing the computation amongst multiple computers in the hopes of gaining
            a speed boost on processing large quantities of data....

            NOTE!
            this code differs from mainDistributed.py by the fact that it does not have as much overhead for getting the
            code to run on multiple cores

            NOTE!
            number of processes count should be properly configured in the main code below

            NOTE!
            this is a bit slower than the dispy code... but the speed boost is still signifigant


'''

import multiprocessing as mp
from TagMediatedEvolution import *
import matplotlib.pyplot as plt
import numpy as np


#Plotting Code: Passed a series list [series1,series2] where series {name:"",x:[],y:[]}
#NOTE! CAN ONLY PLOT 3 colors before it starts using random values for colors
def plot(series,x_name="x",y_name="y",title="Graph"):#Code adapted from my chaotic IFS project
    colors=[(70/255, 240/255, 240/255),(240/255, 50/255, 230/255),(210/255, 245/255, 60/255)]
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    idx = 0
    r = random.random
    for group in series:
        # Plot the points
        plt.scatter(group['x'], group['y'], c=[colors[idx] if idx<len(colors) else (r(),r(),r())],s=np.pi * 3, alpha=0.5, label=group['name'])
        idx += 1
    plt.legend(loc='upper left');
    plt.show()

#the function which computes a single tagMediatedEvolution call... note some are longer compute time than others...
def work(args):
    dataObject = {'x': [], 'y': []}  # store the current data
    #unpack arguments
    g, l, POPULATION_SIZE, STRATEGY_MUTATION_PROB, TAG_MUTATION_PROB,PAYOFF_CONSTANTS = args[0],args[1],args[2],args[3],args[4],args[5]
    tagMediatedEvolution(g, l, POPULATION_SIZE, STRATEGY_MUTATION_PROB, TAG_MUTATION_PROB,PAYOFF_CONSTANTS,log=False,data=dataObject)
    return dataObject

#Main code
if __name__ == "__main__":
    # ================== Settings =======================================================================================================================
    POPULATION_SIZE=100                             #Determines the size of the population
    STRATEGY_MUTATION_PROB=0.01                     #Sets tag mutation rate. note 0.01 = 1%
    TAG_MUTATION_PROB= 0.01                         #Acts per individual tag! equiviliant to 0.01=1% here
    TAG_LENGTHS_TO_COMPUTE = [0,4,32]                 #Creates a series of data for that tag length
    ROUNDS_GENERATIONS = range(200,2000,100)          #Sets how many generations of each quantity are computed
                                                    #    ex. 300 gens are computed here [300,500] two seperate evolutions are computed
                                                    #    one for 300 and one for 500... data added into the same series
    SAMPLES_PER_GEN_COUNT=1                         #For any given generation size how many times that generation count should that be redone....
    PAYOFF_CONSTANTS = [1.9, 1.0, 0.002,0.001]      #Sets the payoff constants for the prisoners dilemma:   [T,R,P,S]  where  T>R>P>S and 2R>T+S>2P
    # ============== (END) Settings ===================================================================================================================

    #Run the calculation
    pool = mp.Pool(processes=4)

    #Stores all the seperate series of data to be plotted with differnet colors and names on the graph
    dataSeries=[]
    for l in TAG_LENGTHS_TO_COMPUTE:
        # this is 1 series to be added to all the series...
        allData = {'x': [], 'y': [], 'name': "Tag Quant={0}".format(l)}
        #========== BUILDS THE arguments
        argList=[]
        for g in ROUNDS_GENERATIONS:
            for s in range(SAMPLES_PER_GEN_COUNT): #Collect multiple samples
                argList.append([g, l, POPULATION_SIZE, STRATEGY_MUTATION_PROB, TAG_MUTATION_PROB,PAYOFF_CONSTANTS])
        #========== END

        #Run the calculation and log progress
        for dataObject in tqdm.tqdm(pool.imap_unordered(work,argList),total=len(argList)):
            allData['x'] = allData['x'] + dataObject['x']
            allData['y'] = allData['y'] + dataObject['y']
        dataSeries.append(allData)

    #Plot the data
    plot(dataSeries,x_name="Rounds",y_name="Collective Payoff",title="Data")