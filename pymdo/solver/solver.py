from typing import Dict, List
from enum import Enum
import logging

from numpy import ndarray, zeros
from numpy.linalg import norm
import matplotlib.pyplot as plt

from pymdo.utils.array_and_dict_utils import verify_dict_1d
from pymdo.core.discipline import Discipline
from pymdo.utils.coupling_structure import get_couplings

LOGGER = logging.getLogger(__name__)


class Solver:
    """
    Base solver class 
    """

    class SolverStatus(Enum):
        NOT_CONVERGED = False
        CONVERGED = True

    def __init__(self, name: str, disciplines: List[Discipline], n_iter_max: int = 15,
                 relax_fact: float = 1.0, tol: float = 0.0001) -> None:
        """
        Initialize the solver.

        Args:
            name (str): The solver name.
            disciplines (List[Discipline]): The list of disciplines to be included.
            n_iter_max (int, optional): Maximum solver iterations. Defaults to 15.
            relax_fact (float, optional): Solver relaxation factor. Defaults to 0.9.
            tol (float, optional): Residual tolerance. Defaults to 0.0001.
        """
        self.name = name
        self.disciplines = disciplines
        self.n_iter_max = n_iter_max
        self.relax_fact = relax_fact
        self.tol = tol

        # Coupling variables
        self.coupling_vars = get_couplings(self.disciplines)

        # Iteration number
        self._iter = 0

        # Convergence status and log
        self._conv_status = self.SolverStatus.NOT_CONVERGED
        self._conv_log: List[float] = []

        # Coupling variables values
        self._old_values: Dict[str, ndarray] = {}
        self._values: Dict[str, ndarray] = {}

    def _single_iteration(self):
        """
        Perform a single solver iteration.
        * self._values should be updated here
        * To be implemented by the subclasses.
        """
        raise NotImplementedError

    def _initialize_couplings(self, initial_coupling_values: Dict[str, ndarray]) -> None:
        """
        Initialize the coupling variable values. If the value for a variable is not provided, 
        the solver will try to initialize it from the disciplines' default inputs. 

        Args:
            initial_coupling_values (Dict[str, ndarray], optional): The provided initial values for the coupling variables.
        """
        if initial_coupling_values is None:
            initial_coupling_values = {}

        default_values = {}
        for disc in self.disciplines:
            default_values.update(disc.get_default_inputs())

        for var in self.coupling_vars:
            if var.name in initial_coupling_values:
                self._values[var.name] = initial_coupling_values[var.name]
            elif var.name in default_values:
                self._values[var.name] = default_values[var.name]

        self._values = verify_dict_1d(self.coupling_vars, self._values)

    def _update_conv(self) -> None:
        """
        Update residual metric log and convergence status.
        """
        # Total residual metric
        total = 0.0

        # Compute the residual for each coupling variable
        for var in self.coupling_vars:
            new, old = self._values[var.name], self._old_values[var.name]
            r = new - old
            total += norm(r) / (1 + norm(new))
        self._conv_log.append(total)

        # Check for convergence
        if total < self.tol:
            self._conv_status = self.SolverStatus.CONVERGED

        # Print iteration number and residual metric
        LOGGER.info(
            f"{self.name} - Iteration: {self._iter} - Residual: {total}")
        if self._conv_status == self.SolverStatus.CONVERGED:
            LOGGER.info(
                f"{self.name} has converged in {self._iter} iterations.")

    def _apply_relaxation(self) -> None:
        """
        Apply under/over relaxation
        """
        for var in self.coupling_vars:
            new, old = self._values[var.name], self._old_values[var.name]
            self._values[var.name] = self.relax_fact * \
                new + (1 - self.relax_fact) * old

    def _terminate_condition(self) -> bool:
        """
        Stop the solver if it is converged, or the max number of iterations is reached

        Returs:
            bool: Whether to stop the solver.
        """
        stop_solver = False
        if self._conv_status == self.SolverStatus.CONVERGED or self._iter >= self.n_iter_max:
            stop_solver = True
        total = self._conv_log[-1] if self._iter > 0 else 0
        if stop_solver == True and total > self.tol:
            LOGGER.warn(
                f"{self.name} has reached the maximum number of iterations: ({self.n_iter_max}), but the residual: ({total}), is above the tolerance: ({self.tol})")
        return stop_solver

    def solve(self, initial_coupling_values: Dict[str, ndarray] = None) -> Dict[str, ndarray]:
        """
        Solve the non-linear system for the coupling variables values.

        Args:
            initial_coupling_values (Dict[str, ndarray], optional): The initial values for the coupling variables. See _initialize_couplings() for more info.

        Returns:
            Dict[str, ndarray]: The coupling variabe values produced by the solver.
        """
        # Reset the solver
        self._iter = 0
        self._conv_log = []
        self._conv_status = self.SolverStatus.NOT_CONVERGED
        self._values, self._old_values = {}, {}

        # Initialize the coupling variables values
        self._initialize_couplings(initial_coupling_values)

        # Perform iterations
        while self._terminate_condition() is False:
            # Update values
            self._old_values, self._values = self._values, {}
            self._single_iteration()
            # Apply the relaxation factor, if needed
            self._apply_relaxation()
            # Update residual
            self._update_conv()
            self._iter += 1
        return self._values

    def plot_convergence(self, show: bool = True, save: bool = False, filename: str = None) -> None:
        """
        Plot the residual metric log.

        Args:
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Where to save the plot. If None, defaults to the name of the solver.
        """
        if not self._conv_log:
            return

        # Create the plot
        fig, ax = plt.subplots(1, 1)
        ax.semilogy([i for i in range(len(self._conv_log))],
                    [total for total in self._conv_log], '-o', color="blue")
        ax.axhline(self.tol, xmin=0, xmax=len(self._conv_log), color="red")
        ax.set_title(f"{self.name} residual metric")
        ax.set_ylabel("Residual metric")
        ax.set_xlabel("Iterations")
        ax.grid()

        # Save/show
        if save:
            if filename is None:
                filename = self.name
            plt.savefig(fname=filename)
        if show:
            plt.show()
