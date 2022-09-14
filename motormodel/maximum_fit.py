import openmdao.api as om
import numpy as np

def discrete_induced_exponential(data, rho):
    actual_max = np.amax(data)
    print("actual max:", actual_max)
    # data[:] -= actual_max
    print(f"data: {data[0:20]}")
    e_rho_f = np.exp(rho * (data / actual_max))
    # e_rho_f = np.exp(rho * data)
    print("e rho f: ", e_rho_f)
    numerator = np.sum(data * e_rho_f, axis=0)
    denominator = np.sum(e_rho_f, axis=0)

    print(f"num: {numerator}")
    print(f"denom: {denominator}")

    induced_exp = numerator / denominator
    return induced_exp

def discrete_induced_exponential_bar(data, rho, ie_bar=None):
    if ie_bar is None:
        if np.ndim(data) == 1:
            ie_bar = 1.0
        elif np.ndim(data) == 2:
            ie_bar = np.ones(data.shape[1])

    actual_max = np.amax(data)
    # print("actual max (bar):", actual_max)
    e_rho_f = np.exp(rho * (data - actual_max))
    # e_rho_f = np.exp(rho * data)
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

    # e_rho_f = np.exp(rho * data)
    data_bar += e_rho_f_bar * rho * e_rho_f

    return induced_exp, data_bar

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

    def compute(self, inputs, outputs):
        data_inputs = [inputs[input] for input in inputs]
        if self.data_stack is None:
            self.data_stack = np.empty([len(data_inputs), data_inputs[0].size])
        np.stack(data_inputs, out=self.data_stack)

        rho = self.options["rho"]
        outputs["data_amplitude"] = discrete_induced_exponential(self.data_stack, rho)

if __name__ == "__main__":
    import unittest

    class TestDIE(unittest.TestCase):
        def test_discrete_induced_exponential(self):
            x = np.linspace(-2, 2, 10);
            data = -x**2 + 1

            max = discrete_induced_exponential(data, 10)

            self.assertAlmostEqual(max, 0.9431504730146061)

        def test_discrete_induced_exponential_bar(self):
            x = np.linspace(-2, 2, 10);
            data = -x**2 + 1

            max, data_bar = discrete_induced_exponential_bar(data, 10)

            self.assertAlmostEqual(max, 0.9431504730146061)

            delta = 1e-7
            data_bar_fd = np.zeros_like(data_bar)
            for i in range(data.size):
                data[i] += delta
                data_bar_fd[i] = (discrete_induced_exponential(data, 10) - max) / delta
                self.assertAlmostEqual(data_bar[i], data_bar_fd[i], places=6)
                data[i] -= delta

        def test_discrete_induced_exponential_matrix(self):
            x = np.linspace(-2, 2, 10);
            data = np.zeros([10, 2])
            data[:, 0] = -x**2 + 1
            data[:, 1] = x**3 + 2

            max = discrete_induced_exponential(data, 10)

            self.assertAlmostEqual(max[0], 0.9431504730146061)
            self.assertAlmostEqual(max[1], 10.0)

        def test_discrete_induced_exponential_bar_matrix(self):
            x = np.linspace(-2, 2, 10);
            data = np.zeros([10, 2])
            data[:, 0] = -x**2 + 1
            data[:, 1] = x**3 + 2

            max, data_bar = discrete_induced_exponential_bar(data, 10)

            self.assertAlmostEqual(max[0], 0.9431504730146061)
            self.assertAlmostEqual(max[1], 10.0)

            delta = 1e-7
            data_bar_fd = np.zeros_like(data_bar)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    data[i,j] += delta
                    data_bar_fd[i,:] += (discrete_induced_exponential(data, 10) - max) / delta
                    self.assertAlmostEqual(data_bar[i,j], data_bar_fd[i,j], places=6)
                    data[i,j] -= delta

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

    unittest.main()