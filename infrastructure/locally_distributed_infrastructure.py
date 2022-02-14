import multiprocessing as mp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from agents.regression_tensorflow import RidgeRegressionTF

from infrastructure.distributed_infrastructure import DistributedInfrastructure

from models.neural_network_tensorflow import get_model



class Monitor(object):
    def __init__(self, node_n_links):
        self._manager = mp.Manager()
        self._shared_resources = self._manager.dict({k:[] for k in node_n_links.keys()})
        self._return_values = self._manager.dict()
        self._read_locks = {n:{nn:mp.Event() for nn in neighbors} for n, neighbors in node_n_links.items()}
        self._write_locks = {n:mp.Event() for n in node_n_links.keys()}
        for n in self._write_locks.keys():
            self._write_locks[n].set()

    
    def write_shared_resource(self, node, resource):
        """
        """
        if not self._write_locks[node].wait(timeout=1):
            for nn in self._read_locks[node].keys():
                self._read_locks[node][nn].set()
        self._shared_resources[node] = resource
        self._write_locks[node].clear()
        for nn in self._read_locks[node].keys():
            self._read_locks[node][nn].set()


    def read_shared_resource(self, node, reading_node):
        """
        """
        if not self._read_locks[node][reading_node].wait(timeout=1):
            self._write_locks[node].set()
        resource = self._shared_resources[node]
        self._read_locks[node][reading_node].clear()
        readers_state = [self._read_locks[node][nn].is_set() for nn in self._read_locks[node].keys()]
        if not all(readers_state):
            self._write_locks[node].set()
        return resource


    def put_return_value(self, node, value):
        """
        """
        self._return_values[node] = value

    
    def get_return_values(self):
        """
        """
        return self._return_values




class LocallyDistributedInfrastructure(DistributedInfrastructure):
    """
    Class which implements the interface of DistributedInfrastructure and 
    simulates a network on a local machine through the creation of multiple processes,
    allowing communication between them by means of Inter-Process-Communication strategies.
    """
    def __init__(self, n_agents, agent_class=RidgeRegressionTF, model_class=get_model):
        super(LocallyDistributedInfrastructure, self).__init__()
        self._network_graph = nx.to_undirected(nx.scale_free_graph(n_agents, seed=1960500))
        self._agent_class = agent_class
        self._model_class = model_class
        self._context = mp.get_context('spawn')


    def _create_agents(self, data_splits, batch_size, lambda_, alpha, alpha_decay):
        """
        """
        self._agents = {}
        for node, data_split in zip(self._network_graph.nodes(), data_splits):
            model = self._model_class(10) #self._model_class(10)
            self._agents[node] = self._agent_class(model, data_split, batch_size, lambda_, alpha, alpha_decay, name=node)


    def _create_communication_links(self):
        """
        """
        node_n_links = {n:list(self._network_graph.neighbors(n)) for n in self._network_graph.nodes()}
        self._monitor = Monitor(node_n_links)
        for node in self._network_graph.nodes():
            self._agents[node].set_endpoint(self._monitor, list(self._network_graph.neighbors(node)))


    def _compute_weight_coefficients(self):
        """
        """
        c = {}
        for node_i in self._network_graph.nodes():

            degree_i = nx.degree(self._network_graph, node_i)
            c[node_i] = {}
            complement_i = 0
            for node_j in nx.neighbors(self._network_graph, node_i):

                degree_j = nx.degree(self._network_graph, node_j)
                c[node_i][node_j] = 1 / (max(degree_i, degree_j) + 1)
                complement_i += c[node_i][node_j]
            c[node_i][node_i] = 1 - complement_i

        return c

    
    def show_network(self):
        """
        """
        nx.draw(self._network_graph)
        plt.show()
                    
    
    def start(self, data_splits, epochs, batch_size, lambda_, alpha, epsilon):
        """
        """
        print('Start distributed training')
        c = self._compute_weight_coefficients()
        alpha_decay = lambda a: a * (1 - epsilon * a)
        self._create_agents(data_splits[0], batch_size=batch_size, lambda_=lambda_, alpha=alpha, alpha_decay=alpha_decay)
        self._create_communication_links()
        processes = []
        for j, agent_name in enumerate(self._agents.keys()):
            p = mp.Process(target=self._agents[agent_name].train, args=(epochs, c[agent_name], nx.number_of_nodes(self._network_graph), data_splits[1][j]))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

        self.plot_agents_metric()
        self.plot_agents_consensus()

        
    def plot_agents_metric(self):
        """
        """
        return_values = self._monitor.get_return_values()
        legend = []
        _ = plt.figure(figsize=(16,10))
        for agent_name in sorted(self._agents.keys()):
            plt.plot(return_values[agent_name][0][5:], linewidth=2)
            legend.append(f'Agent {agent_name}')

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.grid()
        plt.legend(legend)
        plt.title('MSE of the agents')
        plt.show() 


    def plot_agents_consensus(self):
        """
        """
        return_values = self._monitor.get_return_values()
        consensus = np.zeros_like(return_values[0][1])
        _ = plt.figure(figsize=(16,10))
        for agent_name in sorted(self._agents.keys()):
            consensus += np.asarray(return_values[agent_name][1])
        
        consensus = consensus / len(list(self._agents.keys()))
        plt.plot(consensus, linewidth=2)

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Disagreement')
        plt.grid()
        plt.title('Disagreement of the agents')
        plt.show()
    