import openmdao.api as om
import numpy as np

def compute_average(data, average=None):
    should_return = False
    if average is None:
        should_return = True
        if len(data.shape) > 1:
            average = np.empty(data.shape[1])

    if average is not None:
        np.sum(data, axis=0, out=average)
    else:
        average = np.sum(data, axis=0)
    average /= data.shape[0]

    if should_return:
        return average

class AverageComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_pts", types=int)

    def setup(self):
        for i in range(self.options["num_pts"]):
            self.add_input(f"data{i}",
                           shape_by_conn=True,
                           desc=" The data to find the average of")

        self.add_output("data_average",
                        copy_shape="data0",
                        desc=" The point-wise average values")

        self.data_stack = None

    def compute(self, inputs, outputs):
        data_inputs = [inputs[input] for input in inputs]
        if self.data_stack is None:
            self.data_stack = np.empty([len(data_inputs), data_inputs[0].size])
        np.stack(data_inputs, out=self.data_stack)

        compute_average(self.data_stack, average=outputs["data_average"])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        npts = self.options["num_pts"]
        if mode == "fwd":
            if "data_average" in d_outputs:
                for input in d_inputs:
                    d_outputs["data_average"] += d_inputs[input] / npts

        elif mode == "rev":
            if "data_average" in d_outputs:
                for input in d_inputs:
                    d_inputs[input] += d_outputs["data_average"] / npts

if __name__ == "__main__":
    import unittest
    from openmdao.utils.assert_utils import assert_check_partials

    class TestComputeAverage(unittest.TestCase):
        def test_compute_average_scalar(self):
            x = np.linspace(-2, 2, 10);
            data = x**2

            max = compute_average(data)

            self.assertAlmostEqual(max, 1.6296296296296293)

        def test_compute_average_vector(self):
            npts = 10
            x = np.zeros((npts, 2))
            x[:, 0] = np.linspace(-2, 2, npts);
            x[:, 1] = np.linspace(-4, 4, npts);
            data = x**2

            average = np.empty(2)
            compute_average(data, average=average)

            self.assertAlmostEqual(average[0], 1.6296296296296293)
            self.assertAlmostEqual(average[1], 6.518518518518517)

    class TestAverageComp(unittest.TestCase):
        def test_average_comp_scalar(self):
            data0 = 2.0
            data1 = 1.0
            data2 = 4.0

            problem = om.Problem()
            ivc = problem.model.add_subsystem("indeps", om.IndepVarComp(),
                                              promotes_outputs=["*"])
            ivc.add_output("data0", data0)
            ivc.add_output("data1", data1)
            ivc.add_output("data2", data2)

            problem.model.add_subsystem("average",
                                        AverageComp(num_pts=3),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["data_average"])

            problem.setup()
            problem.run_model()

            average = problem.get_val("data_average")
            self.assertAlmostEqual(average, 7/3)

        def test_average_comp_scalar_partials(self):
            data0 = 2.0
            data1 = 1.0
            data2 = 4.0

            problem = om.Problem()
            ivc = problem.model.add_subsystem("indeps", om.IndepVarComp(),
                                              promotes_outputs=["*"])
            ivc.add_output("data0", data0)
            ivc.add_output("data1", data1)
            ivc.add_output("data2", data2)

            problem.model.add_subsystem("average",
                                        AverageComp(num_pts=3),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["data_average"])

            problem.setup()
            problem.run_model()

            average = problem.get_val("data_average")
            self.assertAlmostEqual(average, 7/3)
            data = problem.check_partials(form="central")
            assert_check_partials(data)

        def test_average_comp_vector(self):
            data0 = np.array([-1.0, 2.0, 3.0, 1.0])
            data1 = np.array([0.0, 1.0, 4.0, 1.0])
            data2 = np.array([1.0, 2.0, 3.0, 2.0])

            problem = om.Problem()
            ivc = problem.model.add_subsystem("indeps", om.IndepVarComp(),
                                              promotes_outputs=["*"])
            ivc.add_output("data0", data0)
            ivc.add_output("data1", data1)
            ivc.add_output("data2", data2)

            problem.model.add_subsystem("average",
                                        AverageComp(num_pts=3),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["data_average"])

            problem.setup()
            problem.run_model()

            average = problem.get_val("data_average")
            self.assertAlmostEqual(average[0], 0.0)
            self.assertAlmostEqual(average[1], 5/3)
            self.assertAlmostEqual(average[2], 10/3)
            self.assertAlmostEqual(average[3], 4/3)
            
        def test_average_comp_vector_partials(self):
            data0 = np.array([-1.0, 2.0, 3.0, 1.0])
            data1 = np.array([0.0, 1.0, 4.0, 1.0])
            data2 = np.array([1.0, 2.0, 3.0, 2.0])

            problem = om.Problem()
            ivc = problem.model.add_subsystem("indeps", om.IndepVarComp(),
                                              promotes_outputs=["*"])
            ivc.add_output("data0", data0)
            ivc.add_output("data1", data1)
            ivc.add_output("data2", data2)

            problem.model.add_subsystem("average",
                                        AverageComp(num_pts=3),
                                        promotes_inputs=["*"],
                                        promotes_outputs=["data_average"])

            problem.setup()
            problem.run_model()

            average = problem.get_val("data_average")
            self.assertAlmostEqual(average[0], 0.0)
            self.assertAlmostEqual(average[1], 5/3)
            self.assertAlmostEqual(average[2], 10/3)
            self.assertAlmostEqual(average[3], 4/3)

            data = problem.check_partials(form="central")
            assert_check_partials(data)
    
    unittest.main()