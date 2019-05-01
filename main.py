#!./bin/python
'''
Code By Michael Sherif Naguib
License: MIT
Date: 4/12/19
@University of Tulsa
Description: I have been given the paper:  The Success and Failure of Tag-Mediated Evolution of Cooperation   and asked
             to read the first four pages which does not present yet the solution and was given the synopsis that
             after a critical number of tags on an agent, this would affect how an initial population would converge.
             Based off of the prisoners dilemma: there are two types of individuals cooperators and defectors.... whether
             or not the entire population becomes one or another entirely depends on the number of tags

             The goal of this code is to code this and run tests to see if that is the case (as it obviously is based on
             the paper's premis... and to see how the convergenece values change as population factors are adjusted)
'''


#Imports
import matplotlib.pyplot as plt
import numpy as np
from TagMediatedEvolution import *

#Plotting Code: Passed a series list [series1,series2] where series {name:"",x:[],y:[]}
def plot(series,x_name="x",y_name="y",title="Graph"):#Code adapted from my chaotic IFS project
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    idx = 0
    for group in series:
        # Plot the points
        plt.scatter(group['x'], group['y'], c=[(random.random(),random.random(),random.random())],s=np.pi * 3, alpha=0.5, label=group['name'])
        idx += 1
    plt.legend(loc='upper left');
    plt.show()

#Main code
if __name__ == "__main__":
    #Settings
    POPULATION_SIZE=100               #Determines the size of the population
    STRATEGY_MUTATION_PROB=0.01       #note 0.01 = 1%
    TAG_MUTATION_PROB= 0.01           #Acts per individual tag! equiviliant to 0.01=1% here
    TAG_LENGTHS_TO_COMPUTE = [4,32]   #Creates a series of data for that tag length: so l=3 ==> 2^3 groups...
    ROUNDS_GENERATIONS = range(200,500,50)      #Sets how many generations of each quantity are computed
                                      #    ex. 300 gens are computed here [300,500] two seperate evolutions are computed
                                      #    one for 300 and one for 500... data added into the same series
    SAMPLES_PER_GEN_COUNT=1         # for any given generation size how many times that generation count should that be redone....

    #Run the calculation
    #Stores all the seperate series of data to be plotted with differnet colors and names on the graph
    dataSeries=[]
    for l in TAG_LENGTHS_TO_COMPUTE:
        # this is 1 series to be added to all the series...
        allData = {'x': [], 'y': [], 'name': "Tag Quant={0}".format(l)}
        for g in ROUNDS_GENERATIONS:
            for s in range(SAMPLES_PER_GEN_COUNT): #Collect multiple samples
                dataObject = {'x': [], 'y': []}#store the current data
                result = tagMediatedEvolution(g, l, POPULATION_SIZE, STRATEGY_MUTATION_PROB,TAG_MUTATION_PROB, log=True,data=dataObject)
                #append the current data to the series
                allData['x'] = allData['x'] + dataObject['x']
                allData['y'] = allData['y'] + dataObject['y']
        dataSeries.append(allData)

    #Plot the data
    plot(dataSeries,x_name="Rounds",y_name="Collective Payoff",title="Data")

















