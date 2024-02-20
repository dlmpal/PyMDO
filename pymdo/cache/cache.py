from typing import Dict, List
from enum import Enum
from abc import ABC, abstractmethod

from numpy import ndarray

from pymdo.core.variable import Variable


class CachePolicy(Enum):
    LATEST = "latest"
    FULL = "full"


class Cache(ABC):
    def __init__(self, path: str, input_vars: List[Variable], output_vars: List[Variable], dinput_vars: List[Variable] = None,
                 doutput_vars: List[Variable] = None, policy: CachePolicy = CachePolicy.LATEST, tol: float = 1e-9) -> None:
        """
        Initialize cache.
        """
        self.path = path
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.dinput_vars = dinput_vars
        if self.dinput_vars is None:
            self.dinput_vars = self.input_vars
        self.doutput_vars = doutput_vars
        if self.doutput_vars is None:
            self.doutput_vars = self.output_vars
        self.policy = policy
        self.tol = tol

        # Try to load the cache
        self.from_disk()

    @abstractmethod
    def check_if_entry_exists(self, input_values: Dict[str, ndarray]):
        """ 
        Check if there exists an entry for the given inputs.

        Args:
            inputs (Dict[str, ndarray]): Values for each input variable.
        """
        ...

    @abstractmethod
    def add_entry(self, input_values: Dict[str, ndarray], output_values: Dict[str, ndarray] = None,
                  jac: Dict[str, Dict[str, ndarray]] = None) -> None:
        """ 
        Adds an entry in the cache for the given inputs.
        * If an entry already exists, the existing output/jacobian values 
        are overriden by the ones provided.

        Args:
            inputs (Dict[str, ndarray]): Values for each input variable.
            outputs (Dict[str, ndarray], optional): Values for each output variable. Defaults to None.
            jac (Dict[str, Dict[str, ndarray]], optional): Jacobian. Defaults to None.
        """
        ...

    @abstractmethod
    def load_entry(self, input_values: Dict[str, ndarray]):
        """        
        Load the entry for the given inputs, if it exists.

        Args:
            inputs (Dict[str, ndarray]): Values for each input variable.
        """
        ...

    @abstractmethod
    def from_disk(self):
        """
        Load the cache entries from disk
        """
        ...

    @abstractmethod
    def to_disk(self):
        """
        Save the cache entries to disk
        """
        ...
