"""
This module implements the functionality needed to make a PyMDO OptProblem compatible with SciPy optimizers.  
"""
from typing import Dict, List, Tuple, Callable

from numpy import ndarray, zeros
from scipy.optimize import NonlinearConstraint, Bounds, minimize, OptimizeResult

from pymdo.core.constants import FLOAT_DTYPE
from pymdo.core.variable import Variable
from pymdo.utils.array_and_dict_utils import array_to_dict_1d, dict_to_array_1d
from pymdo.utils.array_and_dict_utils import normalize_dict_1d, denormalize_dict_1d
from pymdo.utils.array_and_dict_utils import get_variable_list_size
from pymdo.opt.opt_problem import OptProblem

SciPyFun = Callable[[ndarray], ndarray]


def wrap_bounds(problem: OptProblem) -> Bounds:
    n_design = get_variable_list_size(problem.design_vars)
    lb_arr = zeros(n_design, FLOAT_DTYPE)
    ub_arr = zeros(n_design, FLOAT_DTYPE)
    keep_feasible_arr = zeros(n_design, bool)
    idx = 0
    for var in problem.design_vars:
        lb, ub, keep_feasible = var.get_bounds_as_array(problem.use_norm)
        lb_arr[idx: idx + var.size] = lb
        ub_arr[idx: idx + var.size] = ub
        keep_feasible_arr[idx: idx + var.size] = keep_feasible
        idx += var.size
    return Bounds(lb_arr, ub_arr, keep_feasible_arr)


def wrap_func_and_jac(problem: OptProblem, var: Variable, scalar_func: bool = False) -> Tuple[SciPyFun, SciPyFun]:
    def func(x: ndarray):
        x = array_to_dict_1d(problem.design_vars, x)
        value = problem.update_values(x, var)
        if scalar_func:
            value = value[0]
        return value

    def jac(x: ndarray):
        x = array_to_dict_1d(problem.design_vars, x)
        jac = problem.update_jac(x, var)
        jac = dict_to_array_1d(problem.design_vars, jac)
        return jac
    return func, jac


def wrap_constraints(problem: OptProblem) -> List[NonlinearConstraint]:
    cons = []
    for con in problem.constraints:
        func,  jac = wrap_func_and_jac(problem, con)
        lb, ub, keep_feasible = con.get_bounds_as_array(problem.use_norm)
        cons.append(NonlinearConstraint(
            func, lb, ub, jac, keep_feasible=keep_feasible))
    return cons


def wrap_scipy(problem: OptProblem):
    bounds, cons = wrap_bounds(problem), wrap_constraints(problem)
    func, jac = wrap_func_and_jac(problem, problem.objective, True)
    return func, jac, bounds, cons


def convert_opt_result_scipy(problem: OptProblem, result: OptimizeResult) -> Dict[str, any]:
    _result = {}
    _result["design_vec"] = array_to_dict_1d(problem.design_vars, result.x)
    if problem.use_norm:
        _result["design_vec"] = denormalize_dict_1d(
            problem.design_vars, _result["design_vec"])
    _result["objective"] = result.fun
    _result["jac"] = array_to_dict_1d(problem.design_vars, result.jac)
    _result["iter"] = result.nit
    _result["message"] = result.message
    _result["converged"] = result.success
    return _result


def execute_scipy(problem: OptProblem, initial_design_vec: Dict[str, ndarray], method: str = "SLSQP", **options) -> Dict[str, any]:
    problem.reset()
    if problem.use_norm:
        initial_design_vec = normalize_dict_1d(
            problem.design_vars, initial_design_vec)
    initial_design_vec = dict_to_array_1d(
        problem.design_vars, initial_design_vec)
    func, jac, bounds, cons = wrap_scipy(problem)
    result = minimize(fun=func, x0=initial_design_vec, jac=jac,
                      bounds=bounds, constraints=cons, method=method, **options)
    result = convert_opt_result_scipy(problem, result)
    return result
