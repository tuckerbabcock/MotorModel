import openmdao.api as om
import numpy as np

def ls_sine_fit(f, t, omega=1.0):
    jac = np.empty([f.shape[0], 3])
    jac[:, 0] = 1.0
    jac[:, 1] = np.sin(omega * t)
    jac[:, 2] = np.cos(omega * t)

    coeffs, _, _, _ = np.linalg.lstsq(jac, f, rcond=None)
    f0 = coeffs[0]
    A = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
    phi = np.arctan2(coeffs[2], coeffs[1])
    fit = np.array([f0, A, phi])
    return fit

def sine_fit(f, t, omega=1.0):
    print(f)
    jac = np.empty([f.shape[0], 3])
    jac[:, 0] = 1.0
    jac[:, 1] = np.sin(omega * t)
    jac[:, 2] = np.cos(omega * t)

    coeffs = np.linalg.solve(jac, f)
    print("coeffs:")
    print(coeffs)
    print()
    f0 = coeffs[0]
    A = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
    phi = np.arctan2(coeffs[2], coeffs[1])
    fit = np.array([f0, A, phi])
    print("fit:")
    print(fit)
    print()
    return fit

def sine_fit_bar(f, t, omega=1.0, fit_bar=None):
    if fit_bar is None:
        fit_bar = np.ones(f.shape)

    jac = np.empty([f.shape[0], 3])
    jac[:, 0] = 1.0
    jac[:, 1] = np.sin(omega * t)
    jac[:, 2] = np.cos(omega * t)

    coeffs = np.linalg.solve(jac, f)
    f0 = coeffs[0]
    A = np.sqrt(coeffs[1]**2 + coeffs[2]**2)
    phi = np.arctan2(coeffs[2], coeffs[1])
    fit = np.array([f0, A, phi])

    if np.ndim(A) == 0:
        fit_jac_T = np.array([[1.0, 0.0, 0.0],
                            [0.0, coeffs[1]/A, -coeffs[2]/(A**2)],
                            [0.0, coeffs[2]/A, coeffs[1]/(A**2)]])
        coeffs_bar = fit_jac_T @ fit_bar
    else:
        fit_jac_T = np.zeros([A.size, 3, 3])
        fit_jac_T[:,0,0] = 1.0
        fit_jac_T[:,1,1] = coeffs[1,:] / A
        fit_jac_T[:,2,1] = coeffs[2,:] / A
        fit_jac_T[:,1,2] = -coeffs[2,:] / (A**2)
        fit_jac_T[:,2,2] = coeffs[1,:] / (A**2)
        coeffs_bar = np.einsum("ikl,li->ki", fit_jac_T, fit_bar)

    coeffs_adjoint = np.linalg.solve(np.transpose(jac), coeffs_bar)
    f_bar = coeffs_adjoint
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

def ls_sine_fit_max(f, t, omega=1.0):
    fit = ls_sine_fit(f, t, omega)
    f0 = fit[0]
    A = fit[1]
    max_val = f0 + A
    return max_val

class PeriodicFitMaximum(om.ExplicitComponent):
    """
    Component that finds a point-wise periodic fit of the form f(theta) = A0 + A1*sin(theta + Phi)
    and calculates the point-wise amplitude field
    """
    def initialize(self):
        self.options.declare("data_size", types=int)
        self.options.declare("frequency", default=1.0)

    def setup(self):
        self.add_input("data", shape_by_conn=True, desc=" The data to fit at all theta values ")
        self.add_input("theta", shape_by_conn=True, desc=" The values of theta for each column of data")

        self.add_output("data_amplitude",
                        shape=self.options["data_size"],
                        desc=" The point-wise amplitude of the fit")

    def compute(self, inputs, outputs):
        # create column stack for improved cache efficiency when iterating over rows
        data = inputs["data"]
        theta = inputs["theta"]
        freq = self.options["frequency"]
        if theta.size == 3:
            outputs["data_amplitude"] = sine_fit_max(data, theta, freq)
        elif theta.size > 3:
            outputs["data_amplitude"] = ls_sine_fit_max(data, theta, 1)
        else:
            raise ValueError("Not enough data available to generate a fit!")


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

            f = np.array([2.0, 1.0, 2.0])
            t = np.array([0.0, np.pi/2, np.pi])
            omega = 1

            max_val, f_bar = sine_fit_max_bar(f, t, omega)

            delta = 1e-8
            f = np.array([2.0+delta, 1.0, 2.0])
            f1_dot = (sine_fit_max(f, t, omega) - max_val) / delta
            f = np.array([2.0, 1.0+delta, 2.0])
            f2_dot = (sine_fit_max(f, t, omega) - max_val) / delta
            f = np.array([2.0, 1.0, 2.0+delta])
            f3_dot = (sine_fit_max(f, t, omega) - max_val) / delta

            self.assertAlmostEqual(f_bar[0], f1_dot)
            self.assertAlmostEqual(f_bar[1], f2_dot)
            self.assertAlmostEqual(f_bar[2], f3_dot)

        def test_sine_fit_max_matrix(self):
            f = np.array([[1.0, 2.0],
                          [1.0, 1.0],
                          [2.0, 2.0]])
            t = np.array([0.0, np.pi/2, np.pi])
            omega = 1

            max_vals = sine_fit_max(f, t, omega)
            self.assertAlmostEqual(max_vals[0], 2.2071067811865475)
            self.assertAlmostEqual(max_vals[1], 3.0)

        def test_sine_fit_max_bar_matrix(self):
            f = np.array([[1.0, 2.0],
                          [1.0, 1.0],
                          [2.0, 2.0]])
            t = np.array([0.0, np.pi/2, np.pi])
            omega = 1

            max_vals, f_bar = sine_fit_max_bar(f, t, omega)

            delta = 1e-8
            f = np.array([[1.0+delta, 2.0],
                          [1.0, 1.0],
                          [2.0, 2.0]])
            f11_dot = (sine_fit_max(f, t, omega) - max_vals) / delta
            f = np.array([[1.0, 2.0],
                          [1.0+delta, 1.0],
                          [2.0, 2.0]])
            f21_dot = (sine_fit_max(f, t, omega) - max_vals) / delta
            f = np.array([[1.0, 2.0],
                          [1.0, 1.0],
                          [2.0+delta, 2.0]])
            f31_dot = (sine_fit_max(f, t, omega) - max_vals) / delta
            f = np.array([[1.0, 2.0+delta],
                          [1.0, 1.0],
                          [2.0, 2.0]])
            f12_dot = (sine_fit_max(f, t, omega) - max_vals) / delta
            f = np.array([[1.0, 2.0],
                          [1.0, 1.0+delta],
                          [2.0, 2.0]])
            f22_dot = (sine_fit_max(f, t, omega) - max_vals) / delta
            f = np.array([[1.0, 2.0],
                          [1.0, 1.0],
                          [2.0, 2.0+delta]])
            f32_dot = (sine_fit_max(f, t, omega) - max_vals) / delta
            self.assertAlmostEqual(f_bar[0,0], f11_dot[0])
            self.assertAlmostEqual(f_bar[1,0], f21_dot[0])
            self.assertAlmostEqual(f_bar[2,0], f31_dot[0])
            self.assertAlmostEqual(f_bar[0,1], f12_dot[1])
            self.assertAlmostEqual(f_bar[1,1], f22_dot[1])
            self.assertAlmostEqual(f_bar[2,1], f32_dot[1])

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

            self.assertAlmostEqual(f_bar[0], f1_dot)
            self.assertAlmostEqual(f_bar[1], f2_dot)
            self.assertAlmostEqual(f_bar[2], f3_dot)
    
    class TestLSSineFit(unittest.TestCase):
        def test_ls_sine_fit_max(self):
            f = np.array([1.0, 1.0, 2.0, 2.0])
            t = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
            omega = 1

            max_val = ls_sine_fit_max(f, t, omega)
            self.assertAlmostEqual(max_val, 2.2071067811865475)

        def test_ls_sine_fit_max_matrix(self):
            f = np.array([[1.0, 2.0],
                          [1.0, 1.0],
                          [2.0, 2.0],
                          [2.0, 3.0]])
            t = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
            omega = 1

            max_vals = ls_sine_fit_max(f, t, omega)
            self.assertAlmostEqual(max_vals[0], 2.2071067811865475)
            self.assertAlmostEqual(max_vals[1], 3.0)
    class TestPeriodicFitMaximum(unittest.TestCase):
        def test_sine_fit_max(self):
            data = np.array([[-1.0, 2.0, 3.0, 1.0],
                             [0.0, 1.0, 4.0, 1.0],
                             [1.0, 2.0, 3.0, 2.0]])

            theta = np.array([0.0, np.pi/2, np.pi])

            problem = om.Problem()
            ivc = problem.model.add_subsystem("indeps", om.IndepVarComp())
            problem.model.add_subsystem("fit", PeriodicFitMaximum(data_size=data.shape[1]))

            ivc.add_output("data", data)
            ivc.add_output("theta", theta)
            problem.model.connect("indeps.data", "fit.data")
            problem.model.connect("indeps.theta", "fit.theta")

            problem.setup()
            problem.run_model()

            data_amp = problem.get_val("fit.data_amplitude")
            print(data_amp)
            self.assertAlmostEqual(data_amp[0], 1.0)
            self.assertAlmostEqual(data_amp[1], 3.0)
            self.assertAlmostEqual(data_amp[2], 4.0)
            self.assertAlmostEqual(data_amp[3], 2.2071067811865475)

        def test_ls_sine_fit_max(self):
            data = np.array([[-1.0, 2.0, 3.0, 1.0],
                             [0.0, 1.0, 4.0, 1.0],
                             [1.0, 2.0, 3.0, 2.0],
                             [0.0, 3.0, 2.0, 2.0]])

            theta = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])

            problem = om.Problem()
            ivc = problem.model.add_subsystem("indeps", om.IndepVarComp())
            problem.model.add_subsystem("fit", PeriodicFitMaximum(data_size=data.shape[1]))

            ivc.add_output("data", data)
            ivc.add_output("theta", theta)
            problem.model.connect("indeps.data", "fit.data")
            problem.model.connect("indeps.theta", "fit.theta")

            problem.setup()
            problem.run_model()

            data_amp = problem.get_val("fit.data_amplitude")
            print(data_amp)
            self.assertAlmostEqual(data_amp[0], 1.0)
            self.assertAlmostEqual(data_amp[1], 3.0)
            self.assertAlmostEqual(data_amp[2], 4.0)
            self.assertAlmostEqual(data_amp[3], 2.2071067811865475)
    unittest.main()
