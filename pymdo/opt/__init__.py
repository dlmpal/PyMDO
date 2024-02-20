from typing import List
from enum import Enum

from pymdo.core.variable import Variable
from pymdo.core.discipline import Discipline
from pymdo.opt.opt_problem import OptProblem
from pymdo.opt.problems.sequential import Sequential
from pymdo.opt.problems.mdf import MDF
from pymdo.opt.problems.idf import IDF


class OptProblemType(str, Enum):
    SEQUENTIAL = "Sequential"
    MDF = "MDF"
    IDF = "IDF"


def create_opt_problem(type: OptProblemType, disciplines: List[Discipline], design_vars: List[Variable], objective: Variable,
                       constraints: List[Variable] = None, maximize_objective: bool = False, use_normalization: bool = True, **options) -> OptProblem:
    kwargs = {"disciplines": disciplines, "design_vars": design_vars, "objective": objective,
              "constraints": constraints, "maximize_objective": maximize_objective, "use_normalization": use_normalization, "name": type}
    kwargs.update(options)
    if type == OptProblemType.SEQUENTIAL:
        return Sequential(**kwargs)
    elif type == OptProblemType.MDF:
        return MDF(**kwargs)
    elif type == OptProblemType.IDF:
        return IDF(**kwargs)
    else:
        raise ValueError(f"OptProblemType {type} is not available.")
