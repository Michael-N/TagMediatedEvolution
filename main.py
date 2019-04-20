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


#imports
import random
import copy
import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np


#Creates an agent:
class Agent:
    def __init__(self,tagQuantity,strategy):
        #Assign properties and  generate random tag...
        self.s=strategy
        # " Additionally, we divide the interval [0, 1] into subintervals of length δ called tag groups" ..
        #  there are 2^ l groups 1/δ =2^l ==> δ = 1/(2^l) ==> sub interval length... so
        self.t = random.randint(0,math.pow(2,tagQuantity))* (1/math.pow(2,tagQuantity))# equivilant to 1/δ * N where n is an integer ==> random tag group...
        self.p=None#Payoff
        self.n=0

    #determines if two agent's tags match: returns true if so, false otherwise
    def tagsMatch(self,otherAgent):
        #Floating point comparison
        return math.isclose(self.t,otherAgent.t,abs_tol=0.00001)
    def __eq__(self, otherAgent):
        return self.tagsMatch(otherAgent)

    #returns an agent that matches this agent's tag else a random one
    def findPartnerAgent(self,population):
        for agent in population:
            if self.tagsMatch(agent):
                return agent
        return population[random.randint(0,len(population)-1)]

    #looks at the strategies of each agent and determines the pay off for the agent calling this method
    def determinePayoff(self,otherAgent):
        #T>R>P>S and 2R>T+S>2P
        #t,r,p,s=1.9,1.0,0.002,0.001
        t, r, p, s = 40,30,20,10
        # Agent A and B: Based on Prisoner's dilemma
        if self.s == otherAgent.s and self.s== 1: #Both cooperate
            return r
        elif self.s == otherAgent.s and self.s == 0: #Both Defect
            return p
        elif self.s==0 and otherAgent.s==1: #A defects B cooperates
            return t
        elif self.s==1 and otherAgent.s==0: #B defected and A cooperated
            return s

    #will randomly mutate strategy and tags based upon the supplied probabilities
    #it may not mutate the Agent at all!
    def randMutate(self,stratProb,tagProb,tagQuantity):
        #if the random number on [0,1] is gtr than the probability then remain same
        self.s = self.s if random.random()>stratProb else random.randint(0,1)
        self.t = self.t if random.random()>tagProb else random.randint(0,math.pow(2,tagQuantity))*(1/math.pow(2,tagQuantity))
        # this is where the old code errored bc the order of rand and tag prob comparison was switched

    #sets the payoff for the agent
    def setPayOff(self,val):
        self.p=val
    #returns a string representaiton of the agent...
    def __str__(self):
        return "(TAG:{0},avePayoff:{1})".format(self.t,self.p)

#preforms universal stochastic sampeling of the population.... see https://en.wikipedia.org/wiki/Stochastic_universal_sampling
def stochasticUniversalSampeling(population,N_offspring):
    # sum the payoffs
    totalPayoffs = sum([a.p for a in population])
    fitnesses = [a.p/totalPayoffs for a in population]
    F = sum(fitnesses)
    P = F/N_offspring
    start = P*random.random()#random number between 0 and p
    pointers = [start+i*P for i in range(0,N_offspring)]

    #roulette wheel selection
    def RWS(population,points):
        keep=[]
        for p in points:
            i=0
            while sum(fitnesses[0:i+1]) < p:
                i+=1
            keep.append(copy.deepcopy(population[i]))
        return keep
    return RWS(population,pointers)

#Plotting Code: Passed a series list [series1,series2] where series {name:"",x:[],y:[]}
def plot(series,x_name="x",y_name="y",title="Graph"):#Code adapted from my chaotic IFS project
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    idx = 0
    for group in series:
        # Plot the points
        plt.scatter(group['x'], group['y'], c=[(random.random(), random.random(), random.random())],s=np.pi * 3, alpha=0.5, label=group['name'])
        idx += 1
    plt.legend(loc='upper left');
    plt.show()

#Tag Mediated Evolution Code! (returns the last generation of agents)
def tagMediatedEvolution(MAX_GENERATIONS,TAG_QUANTITY,POPULATION_SIZE,STRATEGY_MUTATION_PROB,TAG_MUTATION_PROB,log=False,data={'x':[],'y':[]}):
    #Creates a list of agents where half have strategy 0=defect, and half 1=cooperate (NOTE! shuffled)
    population = [(Agent(TAG_QUANTITY,0) if i<(POPULATION_SIZE//2) else Agent(TAG_QUANTITY,1)) for i in range(POPULATION_SIZE)]
    random.shuffle(population)
    #Hales and Edmonds algorithm for evolution with tags
    if log:
        print("Computing Generations:")
    # Basically if do log then use tqdm to log progress else just return the range object
    progressLogger = tqdm.tqdm if log else lambda x:x
    for g in progressLogger(range(MAX_GENERATIONS)):#tqdm.tqdm(range(MAX_GENERATIONS)):
        #calculate Payoffs
        for a in population:
            #assign the payoff of an agent based upon selection and strategies detaied in the paper
            a.setPayOff(a.determinePayoff(a.findPartnerAgent(population)))
        #calculate next generation
        population = stochasticUniversalSampeling(population,len(population))
        for a in population:
            a.randMutate(STRATEGY_MUTATION_PROB,TAG_MUTATION_PROB,TAG_QUANTITY)
        data['x'].append(g)
        data['y'].append(collectivePayoff(population))
    if log:
        print("Generations Complete")
    return population

#Collect population Statistics: Avg Payoff
def collectivePayoff(pop):
    return sum([a.p for a in pop])
#Main code
if __name__ == "__main__":
    #Settings (
    MAX_GENERATIONS=1000        #Limits the number of generations to be computed
    TAG_QUANTITY=32             #specifies the number of binary bits in the tags... used to determine tag groups....
    POPULATION_SIZE=100         #Determines the size of the population
    STRATEGY_MUTATION_PROB=0.1  #equiviliant to 0.01 = 1%
    TAG_MUTATION_PROB= 0.1      #Acts per individual tag! equiviliant to 0.01=1% here

    #dataObject={'x':[],'y':[]}
    #result = tagMediatedEvolution(MAX_GENERATIONS, TAG_QUANTITY, POPULATION_SIZE, STRATEGY_MUTATION_PROB, TAG_MUTATION_PROB,
    #                     PAYOFF_SCALE, log=True, data=dataObject)
    #plot(dataObject['x'],dataObject['y'],x_name="Rounds",y_name="Collective Payoff",title="tag=32")
    series=[]
    for l in [4,32]:
        allData = {'x': [], 'y': [], 'name': "Tag Quant={0}".format(l)}
        for g in [2000]:
            for s in range(1): #Collect multiple samples
                dataObject = {'x': [], 'y': []}
                result = tagMediatedEvolution(g, l, POPULATION_SIZE, STRATEGY_MUTATION_PROB,
                                              TAG_MUTATION_PROB, log=True, data=dataObject)
                allData['x'] = allData['x'] + dataObject['x']
                allData['y'] = allData['y'] + dataObject['y']
        series.append(allData)
    plot(series,x_name="Rounds",y_name="Collective Payoff",title="Data")

















