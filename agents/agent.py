import abc



class Agent(metaclass=abc.ABCMeta):
    """
    Abstract class that defines the common high-level interface for 
    the agents in the distributed infrastructure. 
    """
    def __init__(self, model, data):
        """
        :param model: the model which will be trained in a parallel fashion
        :param data: data split that will be used by the agent for the model's training
        """
        super(Agent, self).__init__()
        self._model = model
        self._data = data
        self._endpoints = None

    def set_endpoint(self, monitor, endpoints_name):
        """
        """
        self._endpoints = sorted(endpoints_name)
        self._monitor = monitor

    @abc.abstractmethod
    def train(**kwargs):
        """
        """
        pass
