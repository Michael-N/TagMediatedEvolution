#!./bin/python
'''
Code By Michael Sherif Naguib
License: MIT
Date: 4/12/19
@University of Tulsa
Description: (Multiprocessing and potential for Distributed computing)

    Please read the description in main.py as it discusses the scope of the project and motivation.
            this code is responsible for distributing the computation amongst multiple computers in the hopes of gaining
            a speed boost on processing large quantities of data.... this is based on the dispy library
'''

import dispy
from TagMediatedEvolution import *
import TagMediatedEvolution
import matplotlib.pyplot as plt
import numpy as np

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

def compute(args):
    try:
        import TagMediatedEvolution
        g, l, POPULATION_SIZE, STRATEGY_MUTATION_PROB, TAG_MUTATION_PROB = args[0],args[1],args[2],args[3],args[4]
        dataObject = {'x': [], 'y': []}  # store the current data
        TagMediatedEvolution.tagMediatedEvolution(g, l, POPULATION_SIZE, STRATEGY_MUTATION_PROB, TAG_MUTATION_PROB, log=False,data=dataObject)
        return dataObject
    except BaseException as e:
        return str(e)
#Main code
if __name__ == "__main__":

    #Settings
    POPULATION_SIZE=100               #Determines the size of the population
    STRATEGY_MUTATION_PROB=0.1       #note 0.01 = 1%
    TAG_MUTATION_PROB= 0.1           #Acts per individual tag! equiviliant to 0.01=1% here
    TAG_LENGTHS_TO_COMPUTE = [4,32]   #Creates a series of data for that tag length
    ROUNDS_GENERATIONS = range(100,500,20)      #Sets how many generations of each quantity are computed
                                      #    ex. 300 gens are computed here [300,500] two seperate evolutions are computed
                                      #    one for 300 and one for 500... data added into the same series
    SAMPLES_PER_GEN_COUNT=1        # for any given generation size how many times that generation count should that be redone....

    #Distributes to the cluster... currently just takes advantage of the entire cpu core count... not networked yet...
    cluster = dispy.JobCluster(compute,depends=[TagMediatedEvolution])
    jobs=[]

    #Run the calculation
    #Stores all the seperate series of data to be plotted with differnet colors and names on the graph
    dataSeries=[]
    for l in TAG_LENGTHS_TO_COMPUTE:
        # this is 1 series to be added to all the series...
        allData = {'x': [], 'y': [], 'name': "Tag Quant={0}".format(l)}
        for g in ROUNDS_GENERATIONS:
            for s in range(SAMPLES_PER_GEN_COUNT): #Collect multiple samples
                job = cluster.submit([g, l, POPULATION_SIZE, STRATEGY_MUTATION_PROB, TAG_MUTATION_PROB])
                jobs.append(job)
        print("All Jobs Submitted: {0}".format(len(jobs)))

        #Get the Results for the first dataset... (Blocking calls)
        for job in tqdm.tqdm(jobs):
            try:
                dataObject = job()
                #print(dataObject)
                # append the current data to the series
                allData['x'] = allData['x'] + dataObject['x']
                allData['y'] = allData['y'] + dataObject['y']
            except BaseException as e:
                print(str(e))

        #Add the data
        dataSeries.append(allData)
    #Print the status of the cluster
    cluster.print_status()

    #Plot the data
    plot(dataSeries,x_name="Rounds",y_name="Collective Payoff",title="Data")
