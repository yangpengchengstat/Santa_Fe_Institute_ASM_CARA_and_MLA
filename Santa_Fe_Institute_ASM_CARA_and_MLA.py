
# coding: utf-8

# In[1]:

import numpy as np
import random, math
import operator
import copy
import matplotlib.pyplot as plt


# Define data type SFI_ASM_with_MLA
# SFI_ASM_with_MLA has methods as follows:
#   1. mean_revertingAORU(self,start_d,T) to generate dividend process
#   
#   2. initialization_Pop(self,prob_match,prob_notmatch) to initialize investors  
#     It returns a dictionary {trader_i:{predictor_j: [12-bit numpy array, forecasting parameters numpy array]}}
#     Each investor has a default forecasting rule where any bit in the 12-bit descriptor is 2, don't care
#     In my code the default rule is the last one, so Pop[i][99] is the default rule of trader i
#     Note the default rule is never eliminated in the GA learning, so we eliminate 20 out 99 rules in GA
#     
#   3. marketCondition(self,price_process,current_d)  
#     After the market updates the price and dividend, this method will be called to generate 12-bit list description of the market state
#     
#   4. match_individual_market_condition(self,individualPredictor,marketcondition) 
#     This method produce on (1) or off(0) for one forecasting rule
#     When we run the code, we use conditionMatch_space to store the return of this method for each rule of each investor, so we get the index     of the active rules.
#     
#   5. fitness(self, individualGene)  
#     The input of this method are all the forecasting rules of one trader and it will calculate the fitness scores for all the 100 rules.
#     When we run the code, given the index of the active rule,  we select the fitness score of the active rule.
#     
#   6. individualDemand(self,E_t,p_t,var_pd_t,cash_last,share_t,d_t,indicator_LA)  
#     This method proces the demand for shares. We consider the max trading amount=10, max short sale=5, cash at hand and share at hand.
#     That indicator_LA is 1 indicates the trder feel losses in the future 
#     
#   7. matingPool(self,individualGene,eliminationNum,prob_cross)  
#     This mmethod performs GA. The 20 of 99 rules are eliminated because the default rule will be always kept. The offspring is generated from 
#     mutation and crossover. Tournament selection is used to choose parent.
#    
#   Note: Each trader's default rule will be updated when GA occurs. 
#         The forecasting parameters (a,b) for this default rule are set to a  weighted average of the values for each of the other rules,
#         where the weight for each rule is one over it's current forecast variance estimate.
#        
#     
#     

# In[2]:

class SFI_ASM_with_MLA:
    
    
    def __init__(self,num_agents,num_predictors,length_predictor,costPerBit,risk_aversion,initial_errorV,totalShare,interest_rate,d_bar,persistence,Var_d,maxTrading,shortSale):
        self.N = num_agents
        self.M = num_predictors
        self.J = length_predictor
        self.lamada = risk_aversion
        self.V_pd = initial_errorV
        self.r = interest_rate
        self.d_bar = d_bar
        self.rho = persistence
        self.sigma = math.sqrt(Var_d)
        self.totalShare = totalShare
        self.c = costPerBit
        self.maxTrading = maxTrading
        self.shortSale = shortSale
        
                
        
    def mean_revertingAORU(self,start_d,T):
        '''The dividend process is exogenously given by a mean-reverting autoregressive ornstein-Uhlenbeck process.
           start_d is the starting point for the process, and T is the number of periods of the generated process.
        '''
        d_space = np.arange(T, dtype=float)
        d_space[0] = start_d #self.d_bar + random.normalvariate(0,self.sigma)
        for i in range(1,T):
            d_space[i] = self.d_bar + self.rho*(d_space[i-1] - self.d_bar) + random.normalvariate(0,self.sigma)
        return d_space
    
        
    def initialization_Pop(self,prob_match,prob_notmatch):
        ''' The condition statement is a J-bit array takes on one of the following three values: 1,0,or 2. 
        1 denotes match, 0 denotes not match, and 2 denotes 'don't care'.
        J = length of predictor
        prob_match = the probability that 1 occurs in one position of J-2 bit condition
        prob_notmatch = the probability that 0 occurs in one position of J-2 bit condition
        the last two bits are the indicators for activation
        prob of 2 = 1-prob_match-prob_notmatch
        forecasting is a list = (forecasting parameter a, forecasting parameter b, price and dividend variance)
        each individual = condition statement + forecasting       
        '''
        prob_notcare = 1.0-prob_match-prob_notmatch
        pop = {i:{j:[] for j in range(self.M)} for i in range(self.N)}
        for i in range(self.N):
            for j in range(self.M-1):
                condition = np.append(np.random.choice([0,1,2], self.J-2, p=[prob_notmatch,prob_match,prob_notcare]), np.array([1,0]))
                forecasing = np.array([random.uniform(0.7,1.2),random.uniform(-10.0,19.0),self.V_pd])
                condition_forecast = [condition,forecasing]
                pop[i][j] = condition_forecast
        for i in range(self.N):
            dontcareCondition = np.append(np.ones(self.J-2,dtype=int)*int(2),np.array([1,0]))
            for j in range(self.M-1):
                abv = 0.0+pop[i][j][1]
            abv_mean = abv/(self.M-1.0)
            dontcareForecasting = np.array([abv_mean[0],abv_mean[1],self.V_pd])
            pop[i][self.M-1]=[dontcareCondition,dontcareForecasting]
        return pop
    
    def marketCondition(self,price_process,current_d):
        price = np.array(price_process)
        pr_over_d = [1.0/4, 1.0/2, 3.0/4, 7.0/8, 1.0, 9.0/8]
        MA = [np.mean(price_process[-i:]) for i in [5,10,100,500]]        
        condition1 = [1 if price[-1]*self.r/current_d>pr_over_d[i] else 0 for i in range(len(pr_over_d))]
        condition2 = [1 if price[-1]>MA[i] else 0 for i in range(len(MA))]
        condition = condition1 + condition2
        return condition
        
    def match_individual_market_condition(self,individualPredictor,marketcondition):
        '''Use this function to check whether the individual cognition of the market state matches the market condition
        The individual predictor has J-2 bit positions in the condition discriptor.
        The condition statement, C, is said to match market state if the following two conditions hold:
        1. C has a 1 or 2in every bit position that market state has 1.
        2. C has a 0 or 2in every bit position that market state has 0'''
        individual_condition = np.array(individualPredictor[:self.J-2])
        market_condition = np.array(marketcondition)
        compare = np.column_stack((individual_condition,market_condition))
        result = np.arange(self.J-2,dtype=int)
        for i in range(self.J-2):
            if compare[i,0] == 2:
                result[i] = 1
            else:
                if compare[i,0] == compare[i,1]:
                    result[i] = 1
                else:
                    result[i] = 0
        if np.sum(result) == self.J-2:
            return individualPredictor[-2]
        else:
            return individualPredictor[-1]
        
    def individualDemand(self,E_t,p_t,var_pd_t,cash_last,share_t,d_t,indicator_LA):
        if indicator_LA==0:
            D_t_unconstraint = (E_t-(1+self.r)*p_t)/(self.lamada*var_pd_t)
        else:
            D_t_unconstraint = (E_t-(1+self.r)*p_t)/(self.lamada*var_pd_t*4.0)
        cash_t=cash_last+share_t*d_t
         
        if D_t_unconstraint>=0:
            if D_t_unconstraint>=share_t:
                if (D_t_unconstraint-share_t)*p_t<=cash_t:
                    if D_t_unconstraint-share_t<=self.maxTrading:
                        buy_t=D_t_unconstraint-share_t
                        Holding_t=buy_t+share_t
                        cashNext=(cash_t-buy_t*p_t)*(1.0+self.r)
                    else:# D_t_unconstraint-share_t>self.maxTrading:
                        buy_t=self.maxTrading
                        Holding_t=self.maxTrading+share_t
                        cashNext=(cash_t-buy_t*p_t)*(1.0+self.r)
                else:# (D_t_unconstraint-share_t)*p_t>cash_t:
                    if cash_t<=self.maxTrading*p_t:
                        buy_t=cash_t/p_t
                        Holding_t=buy_t+share_t
                        cashNext=0.0
                    else:# cash_t/p_t>self.maxTrading:
                        buy_t=self.maxTrading
                        Holding_t=buy_t+share_t
                        cashNext=(cash_t-buy_t*p_t)*(1.0+self.r)
            else:# D_t_unconstraint<share_t:
                if share_t-D_t_unconstraint<=self.maxTrading:
                    sell_t=share_t-D_t_unconstraint
                    Holding_t=share_t-sell_t
                    cashNext=(cash_t+sell_t*p_t)*(1.0+self.r)
                else:# share_t-D_t_unconstraint>self.maxTrading:
                    sell_t=self.maxTrading
                    Holding_t=share_t-sell_t
                    cashNext=(cash_t+sell_t*p_t)*(1.0+self.r)
        else:# D_t_unconstraint<0:
            if share_t>=0:
                if abs(D_t_unconstraint-share_t)<=self.maxTrading:
                    if abs(D_t_unconstraint)<=self.shortSale:
                        sell_t=abs(D_t_unconstraint-share_t)
                        Holding_t=share_t-sell_t
                        cashNext=(cash_t+sell_t*p_t)*(1.0+self.r)
                    else:# abs(D_t_unconstraint)>self.shortSale:
                        sell_t=self.shortSale+share_t
                        Holding_t=share_t-sell_t
                        cashNext=(cash_t+sell_t*p_t)*(1.0+self.r)
                else:# abs(D_t_unconstraint-share_t)>self.maxTrading:
                    if share_t>=self.maxTrading:
                        sell_t=self.maxTrading
                        Holding_t=share_t-sell_t
                        cashNext=(cash_t+sell_t*p_t)*(1.0+self.r)
                    else:
                        largest_shortSell=self.maxTrading-share_t
                        if largest_shortSell<=self.shortSale:
                            sell_t=largest_shortSell+share_t
                            Holding_t=share_t-sell_t
                            cashNext=(cash_t+sell_t*p_t)*(1.0+self.r)
                        else:# largest_shortSell>self.shortSale:
                            sell_t=self.shortSale+share_t
                            Holding_t=share_t-sell_t
                            cashNext=(cash_t+sell_t*p_t)*(1.0+self.r)
            else:# share_t<0:
                if abs(share_t)-abs(D_t_unconstraint)>=0:
                    if (abs(share_t)-abs(D_t_unconstraint))*p_t<=cash_t:
                        Ableto_buy=abs(share_t)-abs(D_t_unconstraint)
                        if Ableto_buy<=self.maxTrading:
                            buy_t=Ableto_buy
                            Holding_t=share_t+buy_t
                            cashNext=(cash_t-buy_t*p_t)*(1.0+self.r)
                        else:# Ableto_buy>self.maxTrading:
                            buy_t=self.maxTrading
                            Holding_t=share_t+buy_t
                            cashNext=(cash_t-buy_t*p_t)*(1.0+self.r)
                    else:# (abs(share_t)-abs(D_t_unconstraint))*p_t>cash_t:
                        largest_buy=cash_t/p_t
                        if largest_buy<=self.maxTrading:
                            buy_t=largest_buy
                            Holding_t=share_t+buy_t
                            cashNext=(cash_t-buy_t*p_t)*(1.0+self.r)
                        else:# largest_buy>self.maxTrading:
                            buy_t=self.maxTrading
                            Holding_t=share_t+buy_t
                            cashNext=(cash_t-buy_t*p_t)*(1.0+self.r)
                else:# abs(share_t)-abs(D_t_unconstraint)<0:
                    Wanto_shortSell=abs(D_t_unconstraint)-abs(share_t)
                    if abs(share_t)<=self.shortSale:
                        if Wanto_shortSell<=self.shortSale-abs(share_t):
                            sell_t=Wanto_shortSell
                            Holding_t=share_t-sell_t
                            cashNext=(cash_t+sell_t*p_t)*(1.0+self.r)
                        else:# Wanto_shortSell>self.shortSale-abs(share_t):
                            sell_t=self.shortSale-abs(share_t)
                            Holding_t=share_t-sell_t
                            cashNext=(cash_t+sell_t*p_t)*(1.0+self.r)
                    else:
                        Holding_t=share_t
                        cashNext=(cash_t)*(1.0+self.r)

        return (Holding_t,cashNext)
    
    
    def fitness(self, individualGene):
        fitness = []        
        for i in range(self.M):
            condition = individualGene[i][0][:-2].copy()
            bitUsed = len(condition[condition!=2])
            fitness.append(-individualGene[i][1][2]-self.c*bitUsed) 
        return fitness
    
    
    def matingPool(self,individualGene,eliminationNum,prob_cross):
        '''The input is the trader's forecasting chromosomes. This function performs making couple,
        crossover, mutation, and returns new gene'''
        fit =  self.fitness(individualGene)        
        fitnessScore = {i:fit[i] for i in range(self.M-1)}
        sorted_fitScore = sorted(fitnessScore.items(), key=operator.itemgetter(1))
        cutoff = int(eliminationNum)
        survivals = sorted_fitScore[cutoff:]
        failed = sorted_fitScore[:cutoff]
        tempt = np.ones(self.M-1-eliminationNum,dtype='float64')
        for i in range(self.M-1-eliminationNum):
            tempt[i] = survivals[i][1]
        #weights = 1.0/tempt
        #p_selection = [weights[i]/sum(weights) for i in range(self.M-1-eliminationNum)]
        candidates = copy.deepcopy(survivals)
        for i in range(eliminationNum):
            toss_crossCoin = np.random.choice([0,1],1,p=[1.0-prob_cross,prob_cross])
            if toss_crossCoin ==0: #mutation
                p_turnamentSelection = [1.0/(len(candidates))]*len(candidates)
                index_parent_candidates = np.random.choice(range(len(candidates)),2,p=p_turnamentSelection)
                parent1_fit = candidates[index_parent_candidates[0]][1]
                parent2_fit = candidates[index_parent_candidates[1]][1]
                if parent1_fit<=parent2_fit:
                    index_parentFormutation_pop = candidates[index_parent_candidates[1]][0]
                    parentFormutation = individualGene[index_parentFormutation_pop]
                    candidates.remove(candidates[index_parent_candidates[1]])
                else:
                    index_parentFormutation_pop = candidates[index_parent_candidates[0]][0]
                    parentFormutation = individualGene[index_parentFormutation_pop]
                    candidates.remove(candidates[index_parent_candidates[0]])
                #choromosome_toMutate=parentFormutation[0].copy()
                for k in range(self.J-2):
                    toss_coin_BitMutation=np.random.choice([0,1],1,p=[1-0.03,0.03])
                    if toss_coin_BitMutation==1:
                        if parentFormutation[0][k]==0:
                            parentFormutation[0][k] = np.random.choice([0,1,2],1,p=[0.0,1.0/3,2.0/3])
                        elif parentFormutation[0][k]==1:
                            parentFormutation[0][k] = np.random.choice([0,1,2],1,p=[1.0/3,0.0,2.0/3])
                        else:
                            parentFormutation[0][k] = np.random.choice([0,1,2],1,p=[1.0/3,1.0/3,1.0/3])
                toss_mutation_parameter_Coin = np.random.choice([0,1,2],1,p=[0.2,0.2,0.6])
                if toss_mutation_parameter_Coin ==0:
                    parentFormutation[1][0],parentFormutation[1][1] = random.uniform(0.7,1.2),random.uniform(-10.,19.0)
                elif toss_mutation_parameter_Coin ==1:
                    parentFormutation[1][0] = parentFormutation[1][0]+random.uniform(-0.0005*(1.2-0.7),0.0005*(1.2-0.7))
                    parentFormutation[1][1] = parentFormutation[1][1]+random.uniform(-0.0005*29,0.0005*29)
                newGene = parentFormutation          
            else: #crossover
                parentForCross={0:[],1:[]}
                for choice in range(2):
                    p_turnamentSelection = [1.0/(len(candidates))]*len(candidates)
                    index_parent_candidates = np.random.choice(range(len(candidates)),2,p=p_turnamentSelection)
                    parent1_fit = candidates[index_parent_candidates[0]][1]
                    parent2_fit = candidates[index_parent_candidates[1]][1]
                    if parent1_fit<=parent2_fit:
                        index_parentFormutation_pop = candidates[index_parent_candidates[1]][0]
                        parentForCross[choice] = individualGene[index_parentFormutation_pop]
                        candidates.remove(candidates[index_parent_candidates[1]])
                    else:
                        index_parentFormutation_pop = candidates[index_parent_candidates[0]][0]
                        parentForCross[choice] = individualGene[index_parentFormutation_pop]
                        candidates.remove(candidates[index_parent_candidates[0]])
                #weight_inverseV=[1.0/parentForCross[i][1][2] for i in range(2)]
                newCondition = []
                for j in range(self.J):
                    tossCoin = np.random.choice([0,1],1,p=[0.5,0.5])
                    newCondition.append(parentForCross[tossCoin[0]][0][j])
                weight_inverseV=[1.0/parentForCross[i][1][2] for i in range(2)]
                newForecast = weight_inverseV[0]/sum(weight_inverseV)*parentForCross[0][1]+weight_inverseV[1]/sum(weight_inverseV)*parentForCross[1][1]
                newGene = [np.array(newCondition),newForecast]
        #put offspring back    
            position_newGene = failed[i][0]
            individualGene[position_newGene] = newGene
        # update the default predictor
        inverseV=np.zeros(self.M-1,dtype=float)
        for j in range(self.M-1):
            inverseV[j] =1.0/individualGene[j][1][2]
        weight_inverseV=inverseV/sum(inverseV)
        a_space = np.zeros(self.M-1,dtype=float)
        b_space = np.zeros(self.M-1,dtype=float)
        V_space = np.zeros(self.M-1,dtype=float)
        for j in range(self.M-1):
            a_space[j]=individualGene[j][1][0]
            b_space[j]=individualGene[j][1][1]
            V_space[j]=individualGene[j][1][2]
        individualGene[self.M-1][1][0] = np.dot(weight_inverseV,a_space)
        individualGene[self.M-1][1][1] = np.dot(weight_inverseV,b_space)
        individualGene[self.M-1][1][2] = np.dot(weight_inverseV,V_space)
        return individualGene


# The following code will:
#   1. initialze the market and the population
#   2. run the loss aversion market and CARA market sparately
#   3. this code is stable for the error of market clearing less than 0.1, i.e. |total bid - total shares in market| <0.1. We set the error of market clearing less than 1 to get the a faster running. I have not tried smaller error, like 0.01, so not sure  if this code could tolerate error <0.01.
#     
#  Note: 
#  
#    1) the actual forecast variance associated with a forecasting rule is calculated and recorded each time a rule is activated, but the specific forecast variance attached to a rule is kept constant between genetic algorithm evolutionary steps and only updated to its actual value during these evolutionary steps. So we clone the investor by Pop_record = copy.deepcopy(Pop) to memorize the actual forecast variance and the initial investor still use the old forecasting rule untill GA occurs.
#    
#    2) totalTime = int(26e4) controls how many periods the market runs. This includes the initial 500 periods when the investor only observed the trading. So in the loop we set t<totalTime -500
#    
#    3) indicator_lossAverse stores Gains or Loss feeling of each trader in each period.
#    
#    4). a_space and b_space store the forecasting parameter for each trader in each perion.
#    
#    5). The subscript _MLA denotes myopic loss aversion. For instance, p_MLA_t is price at t in LA market, p_t is price in CARA market.
# 
# 

# In[3]:

SFIA = SFI_ASM_with_MLA(num_agents=25,num_predictors=100,length_predictor=12,costPerBit=0.005,risk_aversion=0.5,initial_errorV=4.0,totalShare=25,interest_rate=0.1,d_bar=10.0,persistence=0.95,Var_d=0.07429,maxTrading=10.0,shortSale=5.0)

#market

totalTime = int(26e4) # discrete time model
num_traders=25 #there are 25 traders in the market
totalShare = 25 #there are 25 shares of srock in the market

#AR(1) process of dividend
d_bar = 10.0 #mean of d
r = 0.1 #risk-free rate
rho=0.95 #persistence of AR(1)

# price and dividend
D_space = SFIA.mean_revertingAORU(10.0,totalTime) #whole process of d
# there is a initialization periods: 500
D_hist = copy.deepcopy(D_space[:500]) # dividend process of initialization periods
D_MLA_hist = copy.deepcopy(D_space[:500])
P_hist = D_hist/r #pice process of initialization periods
P_MLA_hist = D_MLA_hist/r
D_follow = D_space[500:].copy() # dividend process after initialization periods

#investor
num_preditors=100 # each investor has 100 forecasing rules
lengh_condition=12 # the descriptor has 12 bits
p_learning = 1.0/250 # probability that GA
theta = 1./75# weight on current forecasting error
cash_0 = 2e4*np.ones(num_traders,dtype='float64')[np.newaxis] # initial wealth is 20,000
#cashMLA_0 = 2e4*np.ones(num_traders,dtype='float64')[np.newaxis]
share_0 = np.ones(num_traders,dtype='float64')[np.newaxis] # intial share is 1 for each investor
#shareMLA_0 = np.ones(num_traders,dtype='float64')[np.newaxis]
followingSpace = np.zeros((num_traders,totalTime-1),dtype='float64')
Cash = copy.deepcopy(np.concatenate((cash_0.T,followingSpace),axis=1)) # record of each trader's wealth
Share = copy.deepcopy(np.concatenate((share_0.T,followingSpace),axis=1)) # record of each trader's stock holding
Cash_MLA = copy.deepcopy(np.concatenate((cash_0.T,followingSpace),axis=1))
Share_MLA = copy.deepcopy(np.concatenate((share_0.T,followingSpace),axis=1))
Demand = np.zeros((num_traders,totalTime),dtype='float64') # record of each trader's stock bid/offer
Demand_MLA = np.zeros((num_traders,totalTime),dtype='float64')
E_space = {i:np.zeros(totalTime,dtype='float64') for i in range(num_traders)} # prespace for prediction of mean of future p and d
E_MLA_space = {i:np.zeros(totalTime,dtype='float64') for i in range(num_traders)}
conditionMatch_space = {i:np.array([0]*num_preditors) for i in range(num_traders)}# prespace for condition check
conditionMatch_MLA_space = {i:np.array([0]*num_preditors) for i in range(num_traders)}
index_Best_predictor_space = np.zeros((num_traders,totalTime),dtype='float64')# prespace for index of best predictor
index_MLA_Best_predictor_space = np.zeros((num_traders,totalTime),dtype='float64')
a_space = np.zeros((num_traders,totalTime),dtype='float64')
b_space = np.zeros((num_traders,totalTime),dtype='float64')
V_space = np.zeros((num_traders,totalTime),dtype='float64')
a_MLA_space = np.zeros((num_traders,totalTime),dtype='float64')
b_MLA_space = np.zeros((num_traders,totalTime),dtype='float64')
V_MLA_space = np.zeros((num_traders,totalTime),dtype='float64')
# calculate theoretical price
f=rho/(1+r-rho)
e=(d_bar*(f+d_bar)*(1-rho)-0.5*4)/r
P_theory=f*D_space+e

indicator_lossAverse = np.zeros((num_traders,totalTime),dtype=int)
Pop = SFIA.initialization_Pop(prob_match=0.05,prob_notmatch=0.05)
Pop_MLA=copy.deepcopy(Pop)
Pop_record = copy.deepcopy(Pop)
Pop_MLA_record = copy.deepcopy(Pop_MLA)
t = 0

while t<totalTime-500:
    # At the start of t, d_t publicly posted as D_follow[t]
      # 1) form market state
    market_condition_t = copy.deepcopy(SFIA.marketCondition(P_hist,D_follow[t]))
    market_condition_MLA_t = copy.deepcopy(SFIA.marketCondition(P_MLA_hist,D_follow[t]))
      # 2) match the descriptor and market condition
    for i in range(num_traders):
        for j in range(num_preditors):
            indiv_ij = Pop[i][j][0]
            conditionMatch_space[i][j] = copy.deepcopy(SFIA.match_individual_market_condition(indiv_ij,market_condition_t))  
    for i in range(num_traders):
        for j in range(num_preditors):
            indiv_MLA_ij = Pop_MLA[i][j][0]
            conditionMatch_MLA_space[i][j] = copy.deepcopy(SFIA.match_individual_market_condition(indiv_MLA_ij,market_condition_MLA_t))
    # 3) calculate fitness for each predictor of each trader   
    fitness_space = {i:np.array(SFIA.fitness(Pop[i])) for i in range(num_traders)}
    fitness_MLA_space = {i:np.array(SFIA.fitness(Pop_MLA[i])) for i in range(num_traders)}       
    # 4) choose the best predictor from the active
    for i in range(num_traders):
        index=np.where(conditionMatch_space[i]==1)[0] # index of all active
        active_fit=fitness_space[i][index].copy() # fitness of all active
        where_best_fit=np.argmax(active_fit) # index of best active
        #print(where_best_fit)
        index_Best_predictor_space[i,t]=index[where_best_fit]
    # 5) market maker, auctioneer, starts the auction with price of last period
    for i in range(num_traders):
        index_MLA=np.where(conditionMatch_MLA_space[i]==1)[0] # index of all active
        active_fit_MLA=fitness_MLA_space[i][index_MLA].copy() # fitness of all active
        where_best_fit_MLA=np.argmax(active_fit_MLA) # index of best active
        #print(where_best_fit)
        index_MLA_Best_predictor_space[i,t]=index_MLA[where_best_fit_MLA]
    # 5) market maker, auctioneer, starts the auction with price of last period
    
    startPrice=P_hist[-1]
    startPrice_MLA=P_MLA_hist[-1]
   
    
    p_t=P_hist[-1]
    p_MLA_t=P_MLA_hist[-1]
    largecount=1000  
    diff = 10   
    count=0
    while abs(diff)>1:# diff = total bid - total shares, if abs(diff)<1 or iteration> largecount=1000 (iteration controlled by break)then loop stop
        for i in range(num_traders):
            a_space[i,t],b_space[i,t],V_space[i,t]=Pop[i][index_Best_predictor_space[i,t]][1]#forecasting parameter of the selected rule
                    
            E_space[i][t]=a_space[i,t]*(p_t+D_follow[t])+b_space[i,t]#calulate the expectation of future p and d
            Demand[i][t],Cash[i,t+1]= SFIA.individualDemand(E_space[i][t],p_t,V_space[i,t],Cash[i,t],Share[i,t],D_follow[t],0)#optimal demand
            if Cash[i,t+1]>9e5:
                Cash[i,t+1]=9e5 # avoid explosion of cash because there is exponential growth of cash= cash*(1+r)^t
                Share[i,t+1]=Demand[i][t] #update shares at hand
        totalD=np.sum(Demand[:,t])# calculate total bid
        diff = totalD-totalShare
        valueB_O=abs(diff)
        if count<=largecount:
            if valueB_O>500:  #pricing function: p_trial=p_trial+ c*diff
                p_t+=0.05*diff
            elif 100<valueB_O<=500:
                p_t+=0.02*diff
            elif 80<valueB_O<=100:
                p_t+=0.025*diff
            elif 70<valueB_O<=80:
                p_t+=0.025*diff
            elif 60<valueB_O<=70:
                p_t+=0.025*diff
            elif 50<valueB_O<=60:
                p_t+=0.025*diff
            elif 40<valueB_O<=50:
                p_t+=0.0025*diff
            elif 30<valueB_O<=40:
                p_t+=0.005*diff
            elif 25<valueB_O<=30:
                p_t+=0.004*diff
            elif 10<valueB_O<=25:
                p_t+=0.005*diff
            elif 3<valueB_O<=10:
                p_t+=0.02*diff
            elif 1<valueB_O<=3:
                p_t+=0.005*diff
            elif 0.1<valueB_O<=1:
                p_t+=0.05*diff
            elif 0.01<valueB_O<=0.1:
                p_t+=diff
            elif 0.001<valueB_O<=0.01:
                p_t+=5*diff
            elif 0.0001<valueB_O<=0.001:
                p_t+=50*diff
            elif 0.00001<valueB_O<=0.0001:
                p_t+=500*diff
            else:
                p_t+=500000*diff
        else:
            p_t=D_follow[t]/r
            if t>10000:
                print(t,'CARA traders market******market maker reserved shares =',diff,'at price=', p_t)
            break#if the auction runs for 10000 and no market clearing, break
        count+=1
   

    if t>0:
        for i in range(num_traders):#update forecasting variance for the cloned investor
            a_last,b_last,V_last = Pop_record[i][index_Best_predictor_space[i,t-1]][1]
            V_now = (1.0-theta)*V_last+theta*(a_last*(D_hist[-1]+P_hist[-1])+b_last-p_t-D_follow[t])**2
            if V_now<=500.0:
                Pop_record[i][index_Best_predictor_space[i,t-1]][1][2]=V_now
            else:
                Pop_record[i][index_Best_predictor_space[i,t-1]][1][2]=500.0
              
    for i in range(num_traders):# toss coin to determine if GA occurs
        toss_learningCoin = np.random.choice(range(2),1,p=(1-p_learning,p_learning))
        if toss_learningCoin ==1:
            Pop[i]=copy.deepcopy(Pop_record[i])#copy memory from cloned investor
            Pop[i]=SFIA.matingPool(Pop[i],20,0.1)
    Pop_record = copy.deepcopy(Pop)# clone the investor after GA
   
    D_hist = np.append(D_hist,D_follow[t])
    P_hist = np.append(P_hist,p_t)
        

    #p_MLA_t=startPrice_MLA
    largecount_MLA=1000  
    diff_MLA = 10   
    count_MLA=0
    while abs(diff_MLA)>1:
        for i in range(num_traders):
            a_MLA_space[i,t],b_MLA_space[i,t],V_MLA_space[i,t]=Pop_MLA[i][index_MLA_Best_predictor_space[i,t]][1]
                    
            E_MLA_space[i][t]=a_MLA_space[i,t]*(p_MLA_t+D_follow[t])+b_MLA_space[i,t]
            if E_MLA_space[i][t]-p_MLA_t*(1+r)>=0:
                indicator_lossAverse[i,t]=0
            else:
                indicator_lossAverse[i,t]=1
            Demand_MLA[i][t],Cash_MLA[i,t+1]= SFIA.individualDemand(E_MLA_space[i][t],p_MLA_t,V_MLA_space[i,t],Cash_MLA[i,t],Share_MLA[i,t],D_follow[t],indicator_lossAverse[i,t])
            if Cash_MLA[i,t+1]>9e5:
                Cash_MLA[i,t+1]=9e5
                Share_MLA[i,t+1]=Demand_MLA[i][t]
        totalD_MLA=np.sum(Demand_MLA[:,t])
        diff_MLA = totalD_MLA-totalShare
        valueB_O_MLA=abs(diff_MLA)
        if count_MLA<=largecount_MLA:
            if valueB_O_MLA>100000:
                p_MLA_t+=0.000005*diff_MLA
            elif 50000<valueB_O_MLA<=100000:
                p_MLA_t+=0.0001*diff_MLA
            elif 30000<valueB_O_MLA<=50000:
                p_MLA_t+=0.0002*diff_MLA
            elif 10000<valueB_O_MLA<=30000:
                p_MLA_t+=0.0001*diff_MLA
            elif 5000<valueB_O_MLA<=10000:
                p_MLA_t+=0.001*diff_MLA
            elif 1000<valueB_O_MLA<=5000:
                p_MLA_t+=0.002*diff_MLA
            elif 500<valueB_O_MLA<=1000:
                p_MLA_t+=0.05*diff_MLA
            elif 100<valueB_O_MLA<=500:
                p_MLA_t+=0.02*diff_MLA
            elif 80<valueB_O_MLA<=100:
                p_MLA_t+=0.025*diff_MLA
            elif 70<valueB_O_MLA<=80:
                p_MLA_t+=0.025*diff_MLA
            elif 60<valueB_O_MLA<=70:
                p_MLA_t+=0.025*diff_MLA
            elif 50<valueB_O_MLA<=60:
                p_MLA_t+=0.025*diff_MLA
            elif 40<valueB_O_MLA<=50:
                p_MLA_t+=0.0025*diff_MLA
            elif 30<valueB_O_MLA<=40:
                p_MLA_t+=0.005*diff_MLA
            elif 25<valueB_O_MLA<=30:
                p_MLA_t+=0.004*diff_MLA
            elif 10<valueB_O_MLA<=25:
                p_MLA_t+=0.005*diff_MLA
            elif 3<valueB_O_MLA<=10:
                p_MLA_t+=0.02*diff_MLA
            elif 1<valueB_O_MLA<=3:
                p_MLA_t+=0.005*diff_MLA
            elif 0.1<valueB_O_MLA<=1:
                p_MLA_t+=0.05*diff_MLA
            elif 0.01<valueB_O_MLA<=0.1:
                p_MLA_t+=diff_MLA
            elif 0.001<valueB_O_MLA<=0.01:
                p_MLA_t+=5*diff_MLA
            elif 0.0001<valueB_O_MLA<=0.001:
                p_MLA_t+=50*diff_MLA
            elif 0.00001<valueB_O_MLA<=0.0001:
                p_MLA_t+=500*diff_MLA
            else:
                p_MLA_t+=500000*diff_MLA
        else:
            p_MLA_t=D_follow[t]/r
            if t>10000:
                print(t,'CARA_MLA traders market******market maker reserved shares =',diff_MLA,'at price=', p_MLA_t)
            break
        count_MLA+=1
   
            
    if t>0:
        for i in range(num_traders):
            a_MLA_last,b_MLA_last,V_MLA_last = Pop_MLA_record[i][index_MLA_Best_predictor_space[i,t-1]][1]
            V_MLA_now = (1.0-theta)*V_MLA_last+theta*(a_MLA_last*(D_MLA_hist[-1]+P_MLA_hist[-1])+b_MLA_last-p_MLA_t-D_follow[t])**2
            if V_MLA_now<=500.0:
                Pop_MLA_record[i][index_MLA_Best_predictor_space[i,t-1]][1][2]=V_MLA_now
            else:
                Pop_MLA_record[i][index_MLA_Best_predictor_space[i,t-1]][1][2]=500.0
              
    for i in range(num_traders):
        toss_MLA_learningCoin = np.random.choice(range(2),1,p=(1-p_learning,p_learning))
        if toss_MLA_learningCoin ==1:
            Pop_MLA[i]=copy.deepcopy(Pop_MLA_record[i])
            Pop_MLA[i]=SFIA.matingPool(Pop_MLA[i],20,0.1)
    Pop_MLA_record = copy.deepcopy(Pop_MLA)
   
    D_MLA_hist = np.append(D_MLA_hist,D_follow[t])
    P_MLA_hist = np.append(P_MLA_hist,p_MLA_t)
    
    
    if t%5000==0:
        print('CARA,time is',t,'count',count,'p_CAR is', p_t, 'totalD', totalD,'dif is',diff)
        print('MLA,time is',t,'count',count_MLA,'p_MLA is', p_MLA_t, 'totalD_MLA', totalD_MLA,'dif is',diff_MLA)

    t+=1


# Draw graphs of 200 period with starting period 250,000

# In[4]:

import pandas as pd

starting_t=250000# staring period
t_length=200#time length of the price curve


f=rho/(1+r-rho)
e=(d_bar*(f+1.0)*(1-rho)-0.5*4)/r
P_theory=f*D_space+e

P_RE=P_theory[starting_t:starting_t+t_length].copy()
P_CARA=P_hist[starting_t:starting_t+t_length].copy()
P_LA=P_MLA_hist[starting_t:starting_t+t_length].copy()

dif_RE_CARA=P_RE-P_CARA
dif_RE_LA=P_RE-P_LA
dif_CARA_LA=P_CARA-P_LA

ts_theory = pd.Series(P_RE)
ts_CARA = pd.Series(P_CARA)
ts_LA = pd.Series(P_LA)

ts_dif_RE_CARA=pd.Series(dif_RE_CARA)
ts_dif_RE_LA=pd.Series(dif_RE_LA)

ts_theory.plot(label='Theretical Price')
ts_CARA.plot(label='Simulation of CARA Price')
#ts_dif_RE_CARA.plot(label='difference')
plt.xlabel('period')
plt.ylabel('price')
plt.title('Replication of Fast learning,Figure 1 in LeBaron et al. (1999)')
plt.legend() 
plt.grid(True)
plt.show()




# Draw graphs of 1000 period with starting period 250,000

# In[6]:

starting_t=250000
t_length=1000


f=rho/(1+r-rho)
e=(d_bar*(f+1.0)*(1-rho)-0.5*4)/r
P_theory=f*D_space+e

P_RE=P_theory[starting_t:starting_t+t_length].copy()
P_CARA=P_hist[starting_t:starting_t+t_length].copy()
P_LA=P_MLA_hist[starting_t:starting_t+t_length].copy()

 

ts_theory = pd.Series(P_RE)
ts_CARA = pd.Series(P_CARA)
ts_LA = pd.Series(P_LA)

ts_theory.plot(label='Theretical Price')
ts_CARA.plot(label='Simulation of CARA Price')
ts_LA.plot(label='Simulation of LA Price')
plt.xlabel('period')
plt.ylabel('price')
plt.title('Replication of Fast learning with loss aversion,Figure 1 in LeBaron et al. (1999)')
plt.legend() 
plt.grid(True)
plt.show()


# save data

# In[ ]:


np.save('caraprice', P_hist)
#  P_hist is price history of CARA 


# In[ ]:

np.save('laprice', P_MLA_hist)
#  price history of LA 


# In[ ]:

np.save('dividend', D_space)
# d history


# In[ ]:

np.save('reeprice', P_theory)
# theoretical price


# In[ ]:

# save forecasting parameters
np.save('a_space', a_space)
np.save('b_space', b_space)
np.save('V_space', V_space)
np.save('a_MLA_space', a_MLA_space)
np.save('b_MLA_space',b_MLA_space)
np.save('V_MLA_space',V_MLA_space)
 


# In[ ]:

# save cash and share holding history
np.save('Cash', Cash )
np.save('Share', Share)
np.save('Cash_MLA', Cash_MLA)
np.save('Share_MLA', Share_MLA)
np.save('Demand',Demand)
np.save('Demand_MLA',Demand_MLA)


