from nas.evolution import random_arch, mutate_arch, crossover_arch, conf_onehot
import numpy as np
from tqdm import tqdm

class search(object):
    def __init__(self, accuracy_predictor, latency_predictor, **kwargs):
        self.accuracy_predictor = accuracy_predictor
        self.latency_predictor = latency_predictor

        self.mutate_prob = kwargs.get('mutate_prob', 0.1)            
        self.population_size = kwargs.get('population_size',100) 
        self.num_cycle = kwargs.get('num_cycle',10)
        self.parent_ratio = kwargs.get('parent_ratio',0.25)
        self.crossover_ratio = kwargs.get('crossover_ratio',0.5)
        self.constraint = kwargs.get('constraint')
        
        self.crossover_numbers = int(round(self.crossover_ratio * self.population_size))
        self.parents_size = int(round(self.parent_ratio * self.population_size))
        
    def random_sample(self):
        while True:
            sample = random_arch()  # dictionary[l,c,sz]
            sample_decode = conf_onehot(sample)  #[1,396]
            latency = self.latency_predictor.predict(sample_decode)[0][0]
            if not self.constraint:
                return sample, latency
            elif latency <= self.constraint:
                return sample, latency
            
    def mutate_sample(self, sample):
        while True:
            new_sample = mutate_arch(sample, self.mutate_prob)               
            sample_decode = conf_onehot(new_sample)  #[1,396]
            latency = self.latency_predictor.predict(sample_decode)[0][0]
            if not self.constraint:
                return new_sample, latency
            elif latency <= self.constraint:
                return new_sample, latency
            
    def crossover_sample(self, sample1, sample2):
        while True:
            new_sample = crossover_arch(sample1,sample2)
            sample_decode = conf_onehot(new_sample)  #[1,396]
            latency = self.latency_predictor.predict(sample_decode)[0][0]
            if not self.constraint:
                return new_sample, latency
            elif latency <= self.constraint:
                return new_sample, latency
            
    def evolution_search(self):
        # Generate random population: population is seeded
        child_pool = []
        latency_pool = []    
        for _ in range(self.population_size):
            sample, latency = self.random_sample()
            child_pool.append(sample)
            latency_pool.append(latency)    
        
             
        history = []   
        parents = []
        
        
        for iter in tqdm(range(self.num_cycle)):
            acc_pool = []
            for arch in child_pool:
                arch_decode = conf_onehot(arch)
                acc = self.accuracy_predictor.predict(arch_decode)[0][0]
                acc_pool.append(acc)
            
            population = [] 
            for i in range(len(child_pool)):
                population.append((child_pool[i], acc_pool[i], latency_pool[i]))
                history.append((child_pool[i], acc_pool[i], latency_pool[i]))     
        
            # tournament (combine current population and previous tops)
            population += parents
            
            # evolving the population.(select top 25 among populaton)
            parents = sorted(population, key=lambda x: x[1])[::-1][:self.parents_size]  # according to accuracy
        
               
            child_pool = []
            latency_pool = []    
            
            for _ in range(self.crossover_numbers):
                id1, id2 = np.random.randint(len(parents),size=2)
                if id1 != id2:
                    par_sample1 = parents[id1][0]
                    par_sample2 = parents[id2][0]
                    # Crossover
                    new_sample, latency = self.crossover_sample(par_sample1, par_sample2)
                    child_pool.append(new_sample)
                    latency_pool.append(latency)
        
            mutation_numbers = int(self.population_size - len(child_pool))            
            for _ in range(mutation_numbers):       
                par_sample = parents[np.random.randint(self.parents_size)][0]
                new_sample, latency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                latency_pool.append(latency)
                
        
        acc_pool = []
        for arch in child_pool:
            arch_decode = conf_onehot(arch)
            acc = self.accuracy_predictor.predict(arch_decode)[0][0]
            acc_pool.append(acc)
        
        population = [] 
        for i in range(len(child_pool)):
            population.append((child_pool[i], acc_pool[i], latency_pool[i]))
            history.append((child_pool[i], acc_pool[i], latency_pool[i])) 
        return history                
            
    
class search_multi(search):
    def upfront(self, X, resolution = 20):
        xstep = (np.max(X[:,1])-np.min(X[:,1]))/resolution
        X_xmin = np.min(X[:,1])
        xrange = []
        for i in range(resolution):
            xmin = X_xmin+i*xstep
            xmax = X_xmin+(i+1)*xstep
            xrange.append([xmin,xmax])
            
        xrange = np.array(xrange)
        
        tops = np.zeros((resolution,2))
        p_idx = np.zeros(resolution)
        for i,x in enumerate(xrange):
            for j,s in enumerate(X):
                if s[1] >= x[0] and s[1] <= x[1]:
                    if tops[i][0] ==0:
                        tops[i]=s
                        p_idx[i]=j
                    elif tops[i][0] < s[0]:
                        tops[i]=s
                        p_idx[i]=j
                        
        tops = tops[~np.all(tops == 0, axis=1)]     
        p_idx = p_idx[p_idx != 0]       
        p_idx = p_idx.astype('int')         
        return tops, p_idx

    def evolution_search(self):
        # Generate random population: population is seeded
        child_pool = []
        latency_pool = []    
        for _ in range(self.population_size):
            sample, latency = self.random_sample()
            child_pool.append(sample)
            latency_pool.append(latency)    
        
             
        history = []   
        parents = np.empty([1,3])
        
        
        for iter in tqdm(range(self.num_cycle)):
            acc_pool = []
            for arch in child_pool:
                arch_decode = conf_onehot(arch)
                acc = self.accuracy_predictor.predict(arch_decode)[0][0]
                acc_pool.append(acc)

            
            population = [] 
            scores = []
            for i in range(len(child_pool)):
                population.append((child_pool[i], acc_pool[i], latency_pool[i]))
                scores.append([acc_pool[i],latency_pool[i]])
                history.append((child_pool[i], acc_pool[i], latency_pool[i]))     
        
            # tournament (combine current population and previous tops)
            population = np.array(population)
            population = np.concatenate([population,parents])
            scores = np.array(scores)
            _,p_idx = self.upfront(scores)
            parents = population[p_idx]

                 
            child_pool = []
            latency_pool = []    
            
            for _ in range(self.crossover_numbers):
                id1, id2 = np.random.randint(len(parents),size=2)
                if id1 != id2:
                    par_sample1 = parents[id1][0]
                    par_sample2 = parents[id2][0]
                    # Crossover
                    new_sample, latency = self.crossover_sample(par_sample1, par_sample2)
                    child_pool.append(new_sample)
                    latency_pool.append(latency)
        
            mutation_numbers = int(self.population_size - len(child_pool))            
            for _ in range(mutation_numbers):       
                par_sample = parents[np.random.randint(len(parents))][0]
                new_sample, latency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                latency_pool.append(latency)
                
        
        acc_pool = []
        for arch in child_pool:
            arch_decode = conf_onehot(arch)
            acc = self.accuracy_predictor.predict(arch_decode)[0][0]
            acc_pool.append(acc)
        
        population = [] 
        for i in range(len(child_pool)):
            population.append((child_pool[i], acc_pool[i], latency_pool[i]))
            history.append((child_pool[i], acc_pool[i], latency_pool[i])) 
        return history                