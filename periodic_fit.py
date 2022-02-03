import openmdao.api as om
import numpy as np

def sine_fit(f, t, omega=1.0):
    jac = np.empty([f.shape[0], t.shape[0]])
    jac[:, 0] = 1.0
    jac[:, 1] = np.sin(omega * t)
    jac[:, 2] = np.cos(omega * t)

    coeffs = np.linalg.solve(jac, f)
    f0 = coeffs[0]
    A = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
    phi = np.arctan2(coeffs[2], coeffs[1])
    fit = np.array([f0, A, phi])
    return fit

def sine_fit_bar(f, t, omega=1.0, fit_bar=None):
    if fit_bar is None:
        fit_bar = np.ones(f.shape)

    jac = np.empty([f.shape[0], t.shape[0]])
    jac[:, 0] = 1.0
    jac[:, 1] = np.sin(omega * t)
    jac[:, 2] = np.cos(omega * t)

    coeffs = np.linalg.solve(jac, f)
    f0 = coeffs[0]
    A = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
    phi = np.arctan2(coeffs[2], coeffs[1])
    fit = np.array([f0, A, phi])

    fit_jac_T = np.array([[1.0, 0.0, 0.0],
                        [0.0, coeffs[1]/A, -coeffs[2]/(A**2)],
                        [0.0, coeffs[2]/A, coeffs[1]/(A**2)]])

    coeffs_bar = fit_jac_T @ -fit_bar
    coeffs_adjoint = np.linalg.solve(np.transpose(jac), coeffs_bar)
    f_bar = -coeffs_adjoint
    return fit, f_bar

def sine_fit_avg(f, t, omega=1.0):
    fit = sine_fit(f, t, omega)
    avg = fit[0]
    return avg

def sine_fit_avg_bar(f, t, omega=1.0, avg_bar=1.0):
    fit = sine_fit(f, t, omega)
    avg = fit[0]

    fit_bar = np.zeros(fit.shape)
    fit_bar[0] = avg_bar
    _, f_bar = sine_fit_bar(f, t, omega, fit_bar)
    return avg, f_bar

def sine_fit_max(f, t, omega=1.0):
    fit = sine_fit(f, t, omega)
    f0 = fit[0]
    A = fit[1]
    max_val = f0 + A
    return max_val

def sine_fit_max_bar(f, t, omega=1.0, max_bar=1.0):
    fit = sine_fit(f, t, omega)
    f0 = fit[0]
    A = fit[1]
    max_val = f0 + A

    fit_bar = np.zeros(fit.shape)
    fit_bar[0] = max_bar
    fit_bar[1] = max_bar
    _, f_bar = sine_fit_bar(f, t, omega, fit_bar)
    return max_val, f_bar

class PeriodicFitMaximum(om.ExplicitComponent):
    """
    Component that finds a point-wise periodic fit of the form f(theta) = A + B*sin(theta + Phi)
    and calculates the point-wise amplitude field
    """
    def setup(self):
        self.add_input("data_theta_0", shape_by_conn=True, desc=" The data to fit at theta = 0.0")
        self.add_input("data_theta_pi_2", shape_by_conn=True, desc=" The data to fit at theta = pi/2")
        self.add_input("data_theta_pi", shape_by_conn=True, desc=" The data to fit at theta = pi")

        self.add_output("data_amplitude",
                        shape_by_conn=True,
                        copy_shape="data_theta_0",
                        desc=" The point-wise amplitude of the fit")

    def compute(self, inputs, outputs):
        # create column stack for improved cache efficiency when iterating over rows
        data_stack = np.column_stack((inputs["data_theta_0"],
                                      inputs["data_theta_pi_2"],
                                      inputs["data_theta_pi"]))
        times = np.array([0.0, np.pi/2, np.pi])
        for i in range(data_stack.shape[0]):
            outputs["data_amplitude"][i] = sine_fit_max(data_stack[i, :], times)


if __name__ == "__main__":
    import unittest

    class TestSineFit(unittest.TestCase):
        def test_sine_fit_max(self):
            f = np.array([1.0, 1.0, 2.0])
            t = np.array([0.0, np.pi/2, np.pi])
            omega = 1

            max_val = sine_fit_max(f, t, omega)
            self.assertAlmostEqual(max_val, 2.2071067811865475)

        def test_sine_fit_max_bar(self):
            f = np.array([1.0, 1.0, 2.0])
            t = np.array([0.0, np.pi/2, np.pi])
            omega = 1

            max_val, f_bar = sine_fit_max_bar(f, t, omega)

            delta = 1e-8
            f = np.array([1.0+delta, 1.0, 2.0])
            f1_dot = (sine_fit_max(f, t, omega) - max_val) / delta
            f = np.array([1.0, 1.0+delta, 2.0])
            f2_dot = (sine_fit_max(f, t, omega) - max_val) / delta
            f = np.array([1.0, 1.0, 2.0+delta])
            f3_dot = (sine_fit_max(f, t, omega) - max_val) / delta

            self.assertAlmostEqual(f_bar[0], f1_dot, places=7)
            self.assertAlmostEqual(f_bar[1], f2_dot, places=7)
            self.assertAlmostEqual(f_bar[2], f3_dot, places=7)

        def test_sine_fit_avg(self):
            f = np.array([1.0, 1.0, 2.0])
            t = np.array([0.0, np.pi/2, np.pi])
            omega = 1

            avg_val = sine_fit_avg(f, t, omega)
            self.assertAlmostEqual(avg_val, 1.5)

        def test_sine_fit_avg_bar(self):
            f = np.array([1.0, 1.0, 2.0])
            t = np.array([0.0, np.pi/2, np.pi])
            omega = 1

            avg_val, f_bar = sine_fit_avg_bar(f, t, omega)

            delta = 1e-8
            f = np.array([1.0+delta, 1.0, 2.0])
            f1_dot = (sine_fit_avg(f, t, omega) - avg_val) / delta
            f = np.array([1.0, 1.0+delta, 2.0])
            f2_dot = (sine_fit_avg(f, t, omega) - avg_val) / delta
            f = np.array([1.0, 1.0, 2.0+delta])
            f3_dot = (sine_fit_avg(f, t, omega) - avg_val) / delta

            self.assertAlmostEqual(f_bar[0], f1_dot, places=7)
            self.assertAlmostEqual(f_bar[1], f2_dot, places=7)
            self.assertAlmostEqual(f_bar[2], f3_dot, places=7)
    
    class TestPeriodicFitMaximum(unittest.TestCase):
        def test_ref_motor_current(self):
            problem = om.Problem()
            ivc = problem.model.add_subsystem("indeps", om.IndepVarComp())
            problem.model.add_subsystem("fit", PeriodicFitMaximum())

            data_0 = np.array([-1.0, 2.0, 3.0, 1.0])
            data_pi_2 = np.array([0.0, 1.0, 4.0, 1.0])
            data_pi = np.array([1.0, 2.0, 3.0, 2.0])
            ivc.add_output("data_theta_0", data_0)
            ivc.add_output("data_theta_pi_2", data_pi_2)
            ivc.add_output("data_theta_pi", data_pi)

            problem.model.connect("indeps.data_theta_0", "fit.data_theta_0")
            problem.model.connect("indeps.data_theta_pi_2", "fit.data_theta_pi_2")
            problem.model.connect("indeps.data_theta_pi", "fit.data_theta_pi")

            problem.setup()
            problem.run_model()

            data_amp = problem.get_val("fit.data_amplitude")
            self.assertAlmostEqual(data_amp[0], 1.0)
            self.assertAlmostEqual(data_amp[1], 3.0)
            self.assertAlmostEqual(data_amp[2], 4.0)
            self.assertAlmostEqual(data_amp[3], 2.2071067811865475)

    unittest.main()
