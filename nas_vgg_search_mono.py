# Copyright 2019 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a hands-on tutorial for Hardware-aware Neural Architecture Search with segmentation-based selection (ICICT2022).

we will demonstrate how to get search results using mono-objective optimized search with hardware constraint.
"""
#%% import packages

from keras.models import load_model
from keras import backend as K



#%%  build accuracy predictor

K.set_learning_phase(0)    

# load the full model
accuracy_predictor = load_model('./models/pred_acc_vgg.h5')

#accuracy_predictor.summary()

#%% build latency estimator (FPGA) based on flops
#K.set_learning_phase(0)

latency_predictor = load_model('./models/pred_fpga_latency_vgg.h5')
#latency_predictor = load_model('./models/pred_amd_latency_vgg.h5')
#latency_predictor = load_model('./models/pred_intel_latency_vgg.h5')

#latency_predictor.summary()

#%% Hyper-parameters for the evolutionary search process

mutate_prob = 0.1            
population_size = 100 
num_cycle = 10 
parent_ratio = 0.25
crossover_ratio = 0.5
constraint = 0 # hardware constraint: latency (ms)



hparam = {'mutate_prob': mutate_prob,
          'population_size': population_size,
          'num_cycle': num_cycle,
          'parent_ratio': parent_ratio,
          'crossover_ratio': crossover_ratio,
          'constraint': constraint
          }

#%% evoluton search
""" Please run on CPU
"""
from nas.search import search
nas = search(accuracy_predictor, latency_predictor, **hparam)
history = nas.evolution_search()





