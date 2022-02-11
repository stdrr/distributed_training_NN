import multiprocessing as mp
import networkx as nx
import os

from agents.regression_tensorflow import RidgeRegressionTF

from distributed_infrastructure import DistributedInfrastructure

from models.neural_network_tensorflow import RegressionNN



class LocallyDistributedInfrastructure(DistributedInfrastructure):
    """
    Class which implements the interface of DistributedInfrastructure and 
    simulates a network on a local machine through the creation of multiple processes,
    allowing communication between them by means of Inter-Process-Communication strategies.
    """
    def __init__(self, n_agents):
        super(LocallyDistributedInfrastructure, self).__init__()
        self._network_graph = nx.to_undirected(nx.scale_free_graph(n_agents))
        self._context = mp.get_context('local_network')


    def _create_agents(self, data_splits):
        """
        """
        self._agents = {}
        for node, data_split in zip(self._network_graph.nodes(), data_splits):
            model = RegressionNN(10)
            self._agents[node] = RidgeRegressionTF(model, data_split, 32, 1e-1, 0.0001)


    def _create_communication_links(self):
        """
        """
        edges = self._network_graph.edges()

        for edge in edges:
            conn1, conn2 = mp.Pipe()
            self._agents[edge[0]].set_endpoint(conn1, edge[1])
            self._agents[edge[1]].set_endpoint(conn2, edge[0])


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
                    
    
    def start(self):
        """
        """
        c = self._compute_weight_coefficients()
        data_splits = {}
        self._create_agents(data_splits)
        self._create_communication_links()
        for agent_name in self._agents.keys():
            p = mp.Process(target=self._agents[agent_name].train(epochs=10, c=c))
            p.start()
            p.join()
    