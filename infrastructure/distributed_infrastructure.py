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
        

    @abc.abstractmethod
    def _create_agents(self, data_splits, batch_size, lambda_, alpha):
       """
       """
       pass

    @abc.abstractmethod
    def _create_communication_links(self):
        """
        """
        pass

    @abc.abstractmethod
    def _compute_weight_coefficients(self):
        """
        """
        pass

    @abc.abstractmethod
    def start(self, data_splits, epochs, batch_size, lambda_, alpha):
        """
        """
        pass