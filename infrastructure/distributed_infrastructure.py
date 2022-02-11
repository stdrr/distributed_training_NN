import os
import abc
import networkx as nx




class DistributedInfrastructure(metaclass=abc.ABCMeta):
    """
    Abstract class which provides a high-level abstraction of the distributed
    infrastructure. Subclasses of this class will implement all the methods of its
    interface and might extend its functionalities. 
    """

    def __init__(self):
        super(DistributedInfrastructure, self).__init__()
        self._network_graph = nx.DiGraph()

    @property
    def network_graph(self):
        """
        """
        return self._network_graph

    @abc.abstractmethod
    def _update_graph_abstraction(self):
        """
        """
        pass

    @abc.abstractmethod
    def connect_agent(self, agent, mount_point):
        """
        Given an agent (node), connect it to the infrastructure graph at a given
        mount point.
        """
        pass

    @abc.abstractmethod
    def disconnect_agent(self, agent, mount_point):
        """
        """
        pass