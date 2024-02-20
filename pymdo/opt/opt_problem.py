from typing import Dict, List

from numpy import ndarray, isinf, isneginf

from pymdo.core.variable import Variable
from pymdo.utils.array_and_dict_utils import denormalize_dict_1d, normalize_dict_2d
from pymdo.core.discipline import Discipline

import logging
LOGGER = logging.getLogger(__name__)


class OptProblem(Discipline):
    """
    Base optimization problem class.
    """

    def __init__(self, name: str, disciplines: List[Discipline], design_vars: List[Variable], objective: Variable,
                 constraints: List[Variable] = None, maximize_objective: bool = False, use_normalization: bool = True, **cache_options) -> None:
        """
        Initialize the multidicsiplinary optimization problem.

        Args:
            name (str): The name of the optimization problem.
            disciplines (List[Discipline]): The list of disciplines.
            design_vars (List[Variable]): The list of design variables.
            objective (Variable): The objective.
            constraints: (List[Variable], optional): The list of constraints. A constraint's behaviour is represented by its lower and upper bounds.
            maximize_objective (bool, optional): Whether to maximize the objective. Defaults to False.
            use_normalization (bool, optional): Whether to use normalization. Defaults to True.
        """
        self.disciplines = disciplines
        self.design_vars = design_vars
        self.objective = objective
        self.constraints = constraints
        if self.constraints is None:
            self.constraints = []
        self.max_obj = maximize_objective

        # Normalization can be used only if
        # all design variables have finite bounds
        self.use_norm = use_normalization
        for var in self.design_vars:
            if isinf(var.ub) or isneginf(var.lb):
                self.use_norm = False
                break
            pass

        # Initialize the underlying Discipline
        input_vars = self.design_vars
        output_vars = [self.objective] + self.constraints
        super().__init__(name, input_vars, output_vars,
                         input_vars, output_vars, **cache_options)

        # Optimization log and iteration count
        self._opt_log = []
        self._iter = 0

    def reset(self) -> None:
        """
        Resets the optimization problem.
        Should be called before every execution. 
        """
        self._opt_log, self._iter = [], 0

    def _update_log(self) -> None:
        entry = {}
        entry[self.objective.name] = self._values[self.objective.name].copy()
        self._opt_log.append(entry)

    def update_values(self, design_vec: Dict[str, ndarray], var: Variable) -> ndarray:
        """
        Evaluate the objective function and the constraints for the given design vector.
        Returns the value of the requested variable.

        Args:
            design_vec (Dict[str, ndarray]): The design vector.
            var (Variable): The variable whose value is returned.

        Returns:
            ndarray: The value of the requested variable
        """
        # Deormalize design vector, if needed
        if self.use_norm:
            design_vec = denormalize_dict_1d(self.design_vars, design_vec)

        # Evaluate
        self.eval(design_vec)

        value = self._values[var.name]

        if var.name == self.objective.name:
            LOGGER.info(
                f"{self.name} - Iteration: {self._iter} - Objective: {value[0]}")

            self._update_log()
            self._iter += 1

            # Flip objective sign for maximization
            if self.max_obj:
                value = -value

        return value

    def update_jac(self, design_vec: Dict[str, ndarray], var: Variable) -> Dict[str, ndarray]:
        """
        Differentiate the objective function and the constraints w.r.t the given design vector.
        Returns the jacobian of the requested variable.

        Args:
            design_vec (Dict[str, ndarray]): The design vector.
            var (Variable): The variable whose jacobian is returned.

        Returns:
            Dict[str, ndarray]: The jacobian of the requested variable
        """
        # Denormalize design vector, if needed
        if self.use_norm:
            design_vec = denormalize_dict_1d(self.design_vars, design_vec)

        # Compute jacobian
        self.differentiate(design_vec)

        # Normalize jacobian, if needed
        if self.use_norm:
            self._jac = normalize_dict_2d(
                self.input_vars, self.output_vars, self._jac)

        jac = self._jac[var.name]
        # Flip jacobian sign for maximization
        if self.max_obj and var.name == self.objective.name:
            for var in self.design_vars:
                jac[var.name] = - jac[var.name]
        return jac
