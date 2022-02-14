import os
import tensorflow as tf
import tensorflow_datasets as tfds

from infrastructure.locally_distributed_infrastructure import LocallyDistributedInfrastructure


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

n_agents = 4

# Data
train_split_size, test_split_size = (80 // n_agents, 20 // n_agents)
train_splits = [f'train[{l}%:{u}%]' for l,u, _ in zip(range(0, 81, train_split_size), range(train_split_size, 81, train_split_size), range(n_agents))] 
test_splits = [f'train[{l}%:{u}%]' for l,u, _ in zip(range(80, 101, test_split_size), range(80+test_split_size, 101, test_split_size), range(n_agents))]
train_data_splits = tfds.load('wine_quality', split=train_splits, as_supervised=True)
test_data_splits = tfds.load('wine_quality', split=test_splits, as_supervised=True)

# Distributed training
local_inf = LocallyDistributedInfrastructure(n_agents=n_agents)
local_inf.show_network()
local_inf.start(data_splits=(train_data_splits, test_data_splits), epochs=50, batch_size=32, lambda_=1e-2, alpha=0.00001, epsilon=0.9)