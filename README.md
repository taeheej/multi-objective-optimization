# multi-objective-optimization

Companion code for the ICLR2022 paper: Multi-objective optimization for Hardware-aware
Neural Architecture Search

> **Note:** You do not need a GPU to run this code. Using a modern CPU, you can execute this code within 1 minute.
> **Note:** Some codes are copied from https://github.com/mit-han-lab/once-for-all.

## Running the Code

Install python for your system (v3.6 or later)

Install tensorflow (v1.15.2)

Install keras (2.2.4)

Install matplotlib, tqdm, jupyter

> **Note:** Install tensorboard if the warning message of "Limited tf.compat.v2.summary API due to missing TensorBoard installation" comes out
> **Note:** Install 'h5py==2.10.0' if the error message of "AttributeError: 'str' object has no attribute 'decode'" comes out


## Reproducing Results
To reproduce search results using mono-objective optimized search with hardware constraint, 
we provide tutorial notebook "Hands-on-Tutorial1.ipynb'

To reproduce search results using multi-objective optimization algorithm, 
we provide tutorial notebook "Hands-on_Tutorial2.ipynb'

please clone this repository to your computer and start jupyter notebook to run these tutorial notebooks.

