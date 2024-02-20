from pymdo.solver.solver import Solver
from pymdo.jacobians.jacobian_assembler import JacobianAssembler
from pymdo.opt.opt_problem import OptProblem


class MDF(OptProblem):
    """
    The multidisciplinary-feasible approach for MDO problems.
    """

    def __init__(self, solver: Solver, **kwargs) -> None:
        super().__init__(**kwargs)
        self.solver = solver
        self.assembler = JacobianAssembler()

    def _eval(self):
        # Update the disciplinary inputs according to the values
        # provided by optimizer
        for disc in self.disciplines:
            disc.add_default_inputs(self._values)

        # Solve the system
        self.solver.solve()

        # Grab the values of the constraints and the objective
        # from the disciplinary outputs
        outputs = {}
        for disc in self.disciplines:
            outputs.update(disc.get_output_values())
        for var in self.output_vars:
            self._values[var.name] = outputs[var.name]

    def _differentiate(self) -> None:
        # Evaluate the partials, and then assemble the total derivatives
        partials = {}
        for disc in self.disciplines:
            partials.update(disc.differentiate())
        self._jac = self.assembler.assemble_total(
            self.input_vars, self.output_vars, self.solver.coupling_vars, partials)
