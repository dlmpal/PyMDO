from pymdo.opt.opt_problem import OptProblem


class Sequential(OptProblem):
    """
    Solve an optimization problem sequentially,
    meaning no coupling is enforced between the disciplines.

    * Should only be used for single-discipline problems.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _eval(self) -> None:
        for disc in self.disciplines:
            self._values.update(disc.eval(self._values))

    def _differentiate(self) -> None:
        for disc in self.disciplines:
            self._jac.update(disc.differentiate(self._values))
