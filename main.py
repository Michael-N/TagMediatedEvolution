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

#Creates an agent:
class Agent:
    def __init__(self,tagQuantity,strategy):
        #Assign properties and  generate random tag...
        self.s=strategy
        self.t = [random.randint(0,1) for i in range(tagQuantity)]
        self.p=None#Payoff
        self.n=0

    #determines if two agent's tags match: returns true if so, false otherwise
    def tagsMatch(self,otherAgent):
        for i in range(0,len(self.t)):
            if otherAgent.t[i] != self.t[i]:
                return False
        return True
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
        # Agent A and B: Based on Prisoner's dilemma
        if self.s == otherAgent.s == 1: #Both cooperate
            return 3
        elif self.s == otherAgent.s == 0: #Both Defect
            return 2
        elif self.s==0 and otherAgent.s==1: #A defects B cooperates
            return 4
        else: #B defected and A cooperated
            return 1

    #will randomly mutate strategy and tags based upon the supplied probabilities
    #it may not mutate the Agent at all!
    def randMutate(self,stratProb,tagProb):
        self.s = self.s if random.random()<stratProb else random.randint(0,1)
        self.t = [(tag if random.random()>tagProb else random.randint(0,1))for tag in self.t]

    def addToPayOffAvg(self,val):
        if self.n==0 or self.p==None:
            self.n+=1
            self.p=val
        else:
            self.p = (self.n*self.p + val)/(self.n+1)
            self.n+=1
    def __str__(self):
        return "(TAGS:{0},avePayoff:{1})".format(self.t,self.p)



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

#Main code
if __name__ == "__main__":
    #Settings
    MAX_GENERATIONS=5
    TAG_QUANTITY=2
    POPULATION_SIZE=10
    STRATEGY_MUTATION_PROB=0.00001#equiviliant to 5%
    TAG_MUTATION_PROB= 0.00001# Acts per individual tag! equiviliant to 5% here


    #Creates a list of agents where half have strategy 0=defect, and half 1=cooperate (NOTE! shuffled)
    population = [(Agent(TAG_QUANTITY,0) if i<POPULATION_SIZE//2 else Agent(TAG_QUANTITY,1)) for i in range(POPULATION_SIZE)]
    random.shuffle(population)
    #Hales and Edmonds algorithm for evolution with tags
    print("Computing Generations:")
    for g in range(MAX_GENERATIONS):#tqdm.tqdm(range(MAX_GENERATIONS)):
        #calculate Payoffs
        for a in population:
            print(str(a), end="|\n")
            #assign the payoff of an agent based upon selection and strategies detaied in the paper
            a.addToPayOffAvg(a.determinePayoff(a.findPartnerAgent(population)))
        #calculate next generation
        nextGeneration = stochasticUniversalSampeling(population,len(population))
        for a in nextGeneration:
            a.randMutate(STRATEGY_MUTATION_PROB,TAG_MUTATION_PROB)
        population = nextGeneration
    print("Generations Complete")
    tot_coop = sum([(1 if a.s==1 else 0) for a in population ])
    print("Total Cooperators: {0}\nTotal Defectors: {1}".format(tot_coop,len(population)-tot_coop))








