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

import random
import copy
from keras.utils.np_utils import to_categorical
import numpy as np



def random_arch():
    num_layers = 23   #3*2 + 5*3 + 2 = 23 excluding last prediction layer
    layers = [0, 1] # 0:empty, 1:normal layer(conv or fc), 2:reduction layer(pool)   
    channels = [0.25, 0.5, 0.75, 1] 
    image_sizes = [128, 160, 192, 224, 256]  # multipliers of 32
    L = []
    C = []
    for i in range(num_layers):    
        if i in [0,1,2]:
            C.append(int(random.choice(channels)*64))
        elif i in [3,4,5]:
            C.append(int(random.choice(channels)*128))    
        elif i in [6,7,8,9,10]:
            C.append(int(random.choice(channels)*256))    
        elif i in [11,12,13,14,15]:
            C.append(int(random.choice(channels)*512))    
        elif i in [16,17,18,19,20]:
            C.append(int(random.choice(channels)*512)) 
        elif i in [21,22]:
            C.append(int(random.choice(channels)*4096)) 
            
    for i in range(num_layers):
        if i in [0, 3, 6, 11, 16]:#1st layer as convolution layer in blocks
            L.append(1)
        elif i in [2, 5, 10, 15, 20]:#last layer as pooling layer in blocks
            L.append(2) 
        else:
            L.append(random.choice(layers))           
            
    sample = {
			'l': L,
			'c': C,
            'sz':random.choice(image_sizes)
		}                
    return sample

   
def mutate_arch(sample, mutate_prob):
    '''
    Mutates the input architecture by substitution. The output architecture
       is forced to be different than the input, thus ensuring diversity.
    '''
#    mutate_prob = 0.1  # 0.05 for original
    num_layers = 23
    layers = [0, 1]  # -1:empty, 0:normal layer(conv or fc), 1:reduction layer(pool)     
    channels = [0.25, 0.5, 0.75, 1]
    image_sizes = [128, 160, 192, 224, 256]  # multipliers of 32
    new_sample = copy.deepcopy(sample)
    while True:
        if random.random() < mutate_prob:
            new_sample['sz'] = random.choice(image_sizes)
        
        for i in range(num_layers):
            if random.random() < mutate_prob and i in [0,1,2]:
                new_sample['l'][i] = random.choice(layers)
                new_sample['c'][i] = int(random.choice(channels)*64)
            elif random.random() < mutate_prob and i in [3,4,5]:
                new_sample['l'][i] = random.choice(layers)
                new_sample['c'][i] = int(random.choice(channels)*128)
            elif random.random() < mutate_prob and i in [6,7,8,9,10]:
                new_sample['l'][i] = random.choice(layers)
                new_sample['c'][i] = int(random.choice(channels)*256)
            elif random.random() < mutate_prob and i in [11,12,13,14,15]:
                new_sample['l'][i] = random.choice(layers)
                new_sample['c'][i] = int(random.choice(channels)*512)
            elif random.random() < mutate_prob and i in [16,17,18,19,20]:
                new_sample['l'][i] = random.choice(layers)
                new_sample['c'][i] = int(random.choice(channels)*512)
            elif random.random() < mutate_prob and i in [21,22]:
                new_sample['l'][i] = random.choice(layers)
                new_sample['c'][i] = int(random.choice(channels)*4096)
                
        for i in range(num_layers):
            if i in [0, 3, 6, 11, 16]:#1st layer as convolution layer in blocks
                new_sample['l'][i] = 1                
            elif i in [2, 5, 10, 15, 20]:#last layer as pooling layer in blocks
                new_sample['l'][i] = 2               

        if new_sample == sample:
          continue
        else:
          break
    return new_sample



def crossover_arch(sample1, sample2):
    new_sample = copy.deepcopy(sample1)
    while True:
        for key in new_sample.keys():
            if key != 'sz':
                for i in range(len(new_sample[key])):
                    new_sample[key][i] = random.choice([sample1[key][i], sample2[key][i]])
            elif key == 'sz':
                new_sample[key] = random.choice([sample1[key], sample2[key]])
        if new_sample != sample1:
            break
        else:
            continue        
    return new_sample



#convert all configuration as one-hot encoding
def conf_onehot(X):
    channel = [16,32,48,64,96,128,192,256,384,512,1024,2048,3072,4096]
    channel2id = {16:0,32:1,48:2,64:3,96:4,128:5,192:6,256:7,384:8,512:9,1024:10,2048:11,3072:12,4096:13}
    imagesize = [128, 160, 192, 224, 256]
    size2id = {128:0, 160:1, 192:2, 224:3, 256:4}
    X1l = X['l']
    X1c = X['c']
    X1sz = X['sz']
    
    X1la = to_categorical(X1l,3).astype(np.uint32)
        
    X1cn = np.zeros((len(X1c)))
    for c in channel:
        X1cn[np.array(X1c)== c]=channel2id[c]
    
    X1cna = to_categorical(X1cn,len(channel)).astype(np.uint32)
    
    X1sza = to_categorical(size2id[X1sz],len(imagesize)).astype(np.uint32)
    
    X1lb = np.reshape(X1la,(1,-1))  #(1,69)
    X1cnb = np.reshape(X1cna,(1,-1))  #(1,322)
    X1szb = np.reshape(X1sza, (1,-1)) #(1,5)

    return np.hstack((X1lb,X1cnb,X1szb))  #(1,396)
  
