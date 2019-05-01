#!./bin/python
'''
Code By Michael Sherif Naguib
License: MIT
Date: 4/12/19
@University of Tulsa
Description: this is the code for the component parts of TagMediatedEvolution
'''


#imports
import random
import copy
import tqdm
import math

#Creates an agent:
class Agent:
    def __init__(self,tagLength,strategy,payoffConstants):
        #Assign properties and  generate random tag...
        self.s=strategy
        # " Additionally, we divide the interval [0, 1] into subintervals of length δ called tag groups" ..
        #  there are 2^ l groups 1/δ =2^l ==> δ = 1/(2^l) ==> sub interval length... so
        self.t = random.randint(1,math.pow(2,tagLength))* (1/math.pow(2,tagLength))# equivilant to 1/δ * N where n is an integer ==> random tag group...
        self.p=0#Payoff of the individual
        self.pc = payoffConstants

    #determines if two agent's tags match: returns true if so, false otherwise
    def tagsMatch(self,otherAgent):
        #Floating point comparison
        return math.isclose(self.t,otherAgent.t,abs_tol=0.00000000001)
    def __eq__(self, otherAgent):
        return self.tagsMatch(otherAgent)

    #returns an agent that matches this agent's tag else a random one.
    def findPartnerAgent(self,population):
        random.shuffle(population)#Randomize the order of the population..
        for agent in population:
            if self.tagsMatch(agent):
                return agent
        return population[random.randint(0,len(population)-1)]

    #looks at the strategies of each agent and determines the pay off for the agent calling this method
    def determinePayoff(self,otherAgent):
        #T>R>P>S and 2R>T+S>2P

        t,r,p,s= self.pc[0],self.pc[1],self.pc[2],self.pc[3]
        #t, r, p, s = 4,3,2,1
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
    def randMutate(self,stratProb,tagProb,tagLength):
        #if the random number on [0,1] is gtr than the probability then remain same
        self.s = self.s if random.random()>stratProb else random.randint(0,1)
        self.t = self.t if random.random()>tagProb else random.randint(1,math.pow(2,tagLength))*(1/math.pow(2,tagLength))
        # this is where the old code errored bc the order of rand and tag prob comparison was switched

    #sets the payoff for the agent
    def addToPayOff(self,val):
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

#Tag Mediated Evolution Code! (returns the last generation of agents)
def tagMediatedEvolution(MAX_GENERATIONS,TAG_LENGTH,POPULATION_SIZE,STRATEGY_MUTATION_PROB,TAG_MUTATION_PROB,PAYOFF_CONSTANTS,log=False,data={'x':[],'y':[]}):
    #Creates a list of agents where half have strategy 0=defect, and half 1=cooperate (NOTE! shuffled)
    population = [(Agent(TAG_LENGTH,0,PAYOFF_CONSTANTS) if i<(POPULATION_SIZE//2) else Agent(TAG_LENGTH,1,PAYOFF_CONSTANTS)) for i in range(POPULATION_SIZE)]
    random.shuffle(population)
    #Hales and Edmonds algorithm for evolution with tags
    # Basically if do log then use tqdm to log progress else just return the range object
    progressLogger = tqdm.tqdm if log else lambda x:x
    for g in progressLogger(range(MAX_GENERATIONS)):#tqdm.tqdm(range(MAX_GENERATIONS)):

        #calculate Payoffs
        for a in population:#Each agent only gets one chance at payoff but may interact with others multiple times
            #assign the payoff of an agent based upon selection and strategies detaied in the paper
            a.addToPayOff(a.determinePayoff(a.findPartnerAgent(population)))
        #calculate next generation
        for a in population:
            a.randMutate(STRATEGY_MUTATION_PROB,TAG_MUTATION_PROB,TAG_LENGTH)
        population = stochasticUniversalSampeling(population,len(population))

        #Record the population data
        data['x'].append(g)
        data['y'].append(collectivePayoff(population))

        #Reset the payoffs for the (next) generation
        for a in population:
            a.p=0

        #Shuffle the population
        random.shuffle(population)
    return population

#Collect population Statistics: Avg Payoff
def collectivePayoff(pop):
    return sum([a.p for a in pop])
