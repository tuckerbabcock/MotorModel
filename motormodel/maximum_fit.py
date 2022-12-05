import numpy as np
import openmdao.api as om

def discrete_induced_exponential(data, rho):
    actual_max = np.amax(data)
    e_rho_f = np.exp(rho * (data - actual_max))
    numerator = np.sum(data * e_rho_f, axis=0)
    denominator = np.sum(e_rho_f, axis=0)

    induced_exp = numerator / denominator
    return induced_exp

def discrete_induced_exponential_bar(data, rho, ie_bar=None):
    if ie_bar is None:
        if np.ndim(data) == 1:
            ie_bar = 1.0
        elif np.ndim(data) == 2:
            # compute the gradient of each IE
            ie_bar = np.ones(data.shape[1])

    # print(f"ie_bar: {ie_bar}")
    actual_max = np.amax(data)
    e_rho_f = np.exp(rho * (data - actual_max))
    numerator = np.sum(data * e_rho_f, axis=0)
    denominator = np.sum(e_rho_f, axis=0)

    induced_exp = numerator / denominator

    # reverse pass...
    # induced_exp = numerator / denominator
    num_bar = ie_bar / denominator
    denom_bar = -ie_bar * numerator / (denominator ** 2)

    # denominator = np.sum(e_rho_f, axis=0)
    e_rho_f_bar = np.zeros_like(e_rho_f)
    e_rho_f_bar += denom_bar

    # numerator = np.sum(data * e_rho_f, axis=0)
    e_rho_f_bar += num_bar * data
    data_bar = np.zeros_like(data)
    data_bar += num_bar * e_rho_f

    # e_rho_f = np.exp(rho * (data - actual_max))
    data_bar += e_rho_f_bar * rho * e_rho_f

    return induced_exp, data_bar

def discrete_induced_exponential_dot(data, rho, data_dot):

    actual_max = np.amax(data)
    e_rho_f = np.exp(rho * (data - actual_max))
    e_rho_f_dot = rho * np.exp(rho * (data - actual_max)) * data_dot

    numerator = np.sum(data * e_rho_f, axis=0)
    numerator_dot = np.sum(e_rho_f * data_dot, axis=0) + np.sum(data * e_rho_f_dot, axis=0)

    denominator = np.sum(e_rho_f, axis=0)
    denominator_dot = np.sum(e_rho_f_dot, axis=0)

    induced_exp = numerator / denominator
    induced_exp_dot = numerator_dot / denominator - numerator * denominator_dot/ (denominator ** 2)

    return induced_exp, induced_exp_dot


class DiscreteInducedExponential(om.ExplicitComponent):
    """
    Component that calculates the discrete induced exponential functional to find the maximum
    """
    def initialize(self):
        self.options.declare("num_pts", types=int)
        self.options.declare("rho", default=10.0)

    def setup(self):
        for i in range(self.options["num_pts"]):
            self.add_input(f"data{i}",
                           shape_by_conn=True,
                           desc=" The data to find the maximum of")

        self.add_output("data_amplitude",
                        copy_shape="data0",
                        desc=" The point-wise maximum values",
                        tags=["mphys_coupling"])

        self.data_stack = None
        self.data_dot_stack = None

    def compute(self, inputs, outputs):
        data_inputs = [inputs[input] for input in inputs]
        if self.data_stack is None:
            self.data_stack = np.empty([len(data_inputs), data_inputs[0].size])
        np.stack(data_inputs, out=self.data_stack)

        rho = self.options["rho"]
        outputs["data_amplitude"] = discrete_induced_exponential(self.data_stack, rho)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        data_inputs = [inputs[input] for input in inputs]
        if self.data_stack is None:
            self.data_stack = np.empty([len(data_inputs), data_inputs[0].size])
        np.stack(data_inputs, out=self.data_stack)

        rho = self.options["rho"]
        if mode == "fwd":
            if "data_amplitude" in d_outputs:
                data_dot_inputs = [d_inputs[input] for input in d_inputs]
                if self.data_dot_stack is None:
                    self.data_dot_stack = np.empty([len(data_dot_inputs), data_dot_inputs[0].size])
                np.stack(data_dot_inputs, out=self.data_dot_stack)

                _, die_dot = discrete_induced_exponential_dot(self.data_stack, rho, self.data_dot_stack)
                d_outputs["data_amplitude"] += die_dot

        elif mode == "rev":
            if "data_amplitude" in d_outputs:
                _, data_bar = discrete_induced_exponential_bar(self.data_stack, rho, d_outputs["data_amplitude"])
                for idx, input in enumerate(d_inputs):
                    d_inputs[input] += data_bar[idx, :]

if __name__ == "__main__":
    import unittest
    from openmdao.utils.assert_utils import assert_check_partials

    class TestDIE(unittest.TestCase):
        def test_discrete_induced_exponential(self):
            x = np.linspace(-2, 2, 10);
            data = -x**2 + 1

            max = discrete_induced_exponential(data, 10)

            self.assertAlmostEqual(max, 0.9431504730146061)

        def test_discrete_induced_exponential_bar(self):
            x = np.linspace(-2, 2, 10);
            data = -x**2 + 1

            rho = 10
            _, data_bar = discrete_induced_exponential_bar(data, rho)

            delta = 1e-5
            data_bar_fd = np.zeros_like(data_bar)
            for i in range(data.size):
                data[i] += delta
                data_bar_fd_p = discrete_induced_exponential(data, rho)
                data[i] -= 2*delta
                data_bar_fd_m = discrete_induced_exponential(data, rho)

                data_bar_fd[i] = (data_bar_fd_p - data_bar_fd_m) / (2*delta)
                data[i] += delta

            # print(f"data_bar: {data_bar}")
            # print(f"data_bar_fd: {data_bar_fd}")

            for i in range(data.size):
                self.assertAlmostEqual(data_bar[i], data_bar_fd[i], places=6)

        def test_discrete_induced_exponential_dot(self):
            x = np.linspace(-2, 2, 10);
            data = -x**2 + 1

            rho = 10
            data_dot = np.zeros_like(data)
            delta = 1e-5
            for i in range(data.size):
                data_dot[i] = 1.0
                _, die_dot = discrete_induced_exponential_dot(data, rho, data_dot)
                data_dot[i] = 0.0
                print(f"die_dot: {die_dot}")

                data[i] += delta
                die_dot_fd_p = discrete_induced_exponential(data, rho)
                data[i] -= 2*delta
                die_dot_fd_m = discrete_induced_exponential(data, rho)

                die_dot_fd = (die_dot_fd_p - die_dot_fd_m) / (2*delta)
                print(f"die_dot_fd: {die_dot_fd}")
                data[i] += delta

                self.assertAlmostEqual(die_dot, die_dot_fd, places=6)


        def test_discrete_induced_exponential_matrix(self):
            x = np.linspace(-2, 2, 10);
            data = np.zeros([10, 2])
            data[:, 0] = -x**2 + 1
            data[:, 1] = x**3 + 2

            rho = 10
            max = discrete_induced_exponential(data, rho)

            self.assertAlmostEqual(max[0], 0.9431504730146061)
            self.assertAlmostEqual(max[1], 10.0)

        def test_discrete_induced_exponential_bar_matrix(self):
            x = np.linspace(-2, 2, 10);
            data = np.zeros([10, 2])
            data[:, 0] = -x**2 + 1
            data[:, 1] = x**3 + 2

            rho = 10
            _, data_bar = discrete_induced_exponential_bar(data, rho)

            delta = 1e-5
            data_bar_fd = np.zeros_like(data_bar)
            for i in range(data.shape[0]):
                data[i, :] += delta
                data_bar_fd_p = discrete_induced_exponential(data, rho)
                data[i, :] -= 2*delta
                data_bar_fd_m = discrete_induced_exponential(data, rho)

                data_bar_fd[i, :] = (data_bar_fd_p - data_bar_fd_m) / (2*delta)
                data[i, :] += delta

            # print(f"data_bar: {data_bar}")
            # print(f"data_bar_fd: {data_bar_fd}")

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self.assertAlmostEqual(data_bar[i, j], data_bar_fd[i, j], places=6)

        def test_discrete_induced_exponential_dot_matrix(self):
            x = np.linspace(-2, 2, 10);
            data = np.zeros([10, 2])
            data[:, 0] = -x**2 + 1
            data[:, 1] = x**3 + 2

            rho = 10
            data_dot = np.zeros_like(data)
            delta = 1e-5
            for i in range(data.shape[0]):
                data_dot[i, :] = 1.0
                print(f"data_dot: {data_dot}")
                _, die_dot = discrete_induced_exponential_dot(data, rho, data_dot)
                data_dot[i, :] = 0.0

                data[i, :] += delta
                data_bar_fd_p = discrete_induced_exponential(data, rho)
                data[i, :] -= 2*delta
                data_bar_fd_m = discrete_induced_exponential(data, rho)

                data_bar_fd = (data_bar_fd_p - data_bar_fd_m) / (2*delta)
                data[i, :] += delta

                print(f"die_dot: {die_dot}")
                print(f"die_dot_fd: {data_bar_fd}")

                for j in range(data.shape[1]):
                    self.assertAlmostEqual(die_dot[j], data_bar_fd[j], places=6)

    class TestDiscreteInducedExponential(unittest.TestCase):
        def test_discrete_induced_exponential(self):
            # data = np.array([[-1.0, 2.0, 3.0, 1.0],
            #                  [0.0, 1.0, 4.0, 1.0],
            #                  [1.0, 2.0, 3.0, 2.0]])

            data0 = np.array([-1.0, 2.0, 3.0, 1.0])
            data1 = np.array([0.0, 1.0, 4.0, 1.0])
            data2 = np.array([1.0, 2.0, 3.0, 2.0])

            problem = om.Problem()
            ivc = problem.model.add_subsystem("indeps", om.IndepVarComp(),
                                              promotes_outputs=["*"])
            ivc.add_output("data0", data0)
            ivc.add_output("data1", data1)
            ivc.add_output("data2", data2)

            problem.model.add_subsystem("fit",
                                        DiscreteInducedExponential(num_pts=3),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["data_amplitude"])

            problem.setup()
            problem.run_model()

            data_amp = problem.get_val("data_amplitude")
            # print(data_amp)
            self.assertAlmostEqual(data_amp[0], 0.9999545980092709)
            self.assertAlmostEqual(data_amp[1], 1.9999773005503956)
            self.assertAlmostEqual(data_amp[2], 3.99990920838434)
            self.assertAlmostEqual(data_amp[3], 1.999909208384341)

        def test_discrete_induced_exponential_derivs(self):
            data0 = np.array([-1.0, 2.0, 3.0, 1.0])
            data1 = np.array([0.0, 1.0, 4.0, 1.0])
            data2 = np.array([1.0, 2.0, 3.0, 2.0])

            problem = om.Problem()
            ivc = problem.model.add_subsystem("indeps", om.IndepVarComp(),
                                              promotes_outputs=["*"])
            ivc.add_output("data0", data0)
            ivc.add_output("data1", data1)
            ivc.add_output("data2", data2)

            problem.model.add_subsystem("fit",
                                        DiscreteInducedExponential(num_pts=3),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["data_amplitude"])

            problem.setup()
            problem.run_model()

            data = problem.check_partials(form="central")
            assert_check_partials(data)

    unittest.main()