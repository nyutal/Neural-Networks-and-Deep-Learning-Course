# Created by @nyutal on 21/05/2020
import numpy as np
import pytest
from course_1.week_4.model_layer import Linear, Stacked, Sigmoid, Relu
from course_1.week_4.cost import CrossEntropy
from course_1.week_4.grads import Grads


@pytest.fixture()
def linear():
    return Linear(3, 1)


@pytest.fixture()
def stacked():
    return Stacked([Linear(3, 2), Linear(2, 1)], to_cache=True)


@pytest.fixture()
def deep_stacked():
    return Stacked([Linear(5, 4), Linear(4, 3)], to_cache=True)


@pytest.fixture()
def linear_sigmoid():
    return Stacked([Linear(2, 3), Sigmoid(to_cache=True)], to_cache=True)


@pytest.fixture()
def linear_relu():
    return Stacked([Linear(2, 3), Relu(to_cache=True)], to_cache=True)


def deep_equality(t1, t2):
    assert type(t1) == type(t2)
    if any(isinstance(t1, array_type) for array_type in [tuple, list]):
        assert (len(t1) == len(t2))
        assert all(deep_equality(st1, st2) for (st1, st2) in zip(t1, t2))
    elif isinstance(t1, np.ndarray):
        assert np.allclose(t1, t2)
    else:
        assert (t1 == t2)
    return True


class TestWeek4:
    def test_init_params(self, stacked: Stacked):
        expected = [(np.array([[0.01624345, -0.00611756, -0.00528172], [-0.01072969, 0.00865408, -0.02301539]]),
                     np.array([[0.], [0.]])),
                    (np.array([[0.01744812, -0.00761207]]),
                     np.array([[0.]]))]
        stacked.initialize_parameters(seed=1)
        params = stacked.extract_params()
        assert deep_equality(params, expected)
    
    def test_deep_init_params(self, deep_stacked: Stacked):
        expected = [
            (np.array([[0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
                       [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
                       [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
                       [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]]),
             np.array([[0.],
                       [0.],
                       [0.],
                       [0.]])),
            (np.array([[-0.01185047, -0.0020565, 0.01486148, 0.00236716],
                       [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
                       [-0.00768836, -0.00230031, 0.00745056, 0.01976111]]),
             np.array([[0.],
                       [0.],
                       [0.]]))
        ]
        deep_stacked.initialize_parameters(seed=3)
        params = deep_stacked.extract_params()
        assert deep_equality(params, expected)
    
    def test_forward(self, linear: Linear):
        expected = np.array([[3.26295337, -1.23429987]])
        x = np.array([[1.62434536, -0.61175641],
                      [-0.52817175, -1.07296862],
                      [0.86540763, -2.3015387]])
        w = np.array([[1.74481176, -0.7612069, 0.3190391]])
        b = [[-0.24937038]]
        linear._pack_params(w, b)
        res = linear.forward(x)
        assert (deep_equality(expected, res))
    
    def test_linear_activation_forward(self, linear_sigmoid: Stacked, linear_relu: Stacked):
        a_prev = np.array([[-0.41675785, -0.05626683],
                           [-2.1361961, 1.64027081],
                           [-1.79343559, -0.84174737]])
        w = np.array([[0.50288142, -1.24528809, -1.05795222]])
        b = np.array([[-0.90900761]])
        expected_sigmoid = np.array([[0.96890023, 0.11013289]])
        expected_relu = np.array([[3.43896131, 0.]])
        
        linear_sigmoid._layers[0]._pack_params(w, b)
        assert (deep_equality(linear_sigmoid.forward(a_prev), expected_sigmoid))
        linear_relu._layers[0]._pack_params(w, b)
        assert (deep_equality(linear_relu.forward(a_prev), expected_relu))
    
    def test_deep_linear_activation_forward(self):
        W1 = np.array([[0.35480861, 1.81259031, -1.3564758, -0.46363197, 0.82465384],
                       [-1.17643148, 1.56448966, 0.71270509, -0.1810066, 0.53419953],
                       [-0.58661296, -1.48185327, 0.85724762, 0.94309899, 0.11444143],
                       [-0.02195668, -2.12714455, -0.83440747, -0.46550831, 0.23371059]])
        b1 = np.array([[1.38503523],
                       [-0.51962709],
                       [-0.78015214],
                       [0.95560959]])
        W2 = np.array([[-0.12673638, -1.36861282, 1.21848065, -0.85750144],
                       [-0.56147088, -1.0335199, 0.35877096, 1.07368134],
                       [-0.37550472, 0.39636757, -0.47144628, 2.33660781]])
        b2 = np.array([[1.50278553],
                       [-0.59545972],
                       [0.52834106]])
        W3 = np.array([[0.9398248, 0.42628539, -0.75815703]])
        b3 = np.array([[-0.16236698]])
        
        x = np.array([[-0.31178367, 0.72900392, 0.21782079, -0.8990918],
                      [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
                      [1.63929108, -0.4298936, 2.63128056, 0.60182225],
                      [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
                      [0.07612761, -0.15512816, 0.63422534, 0.810655]])
        
        expected_output = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
        
        model = Stacked(
            [Linear(4, 5),
             Relu(to_cache=True),
             Linear(4, 3),
             Relu(to_cache=True),
             Linear(3, 1),
             Sigmoid(to_cache=True), ],
            to_cache=True)
        
        model._layers[0]._pack_params(W1, b1)
        model._layers[2]._pack_params(W2, b2)
        model._layers[4]._pack_params(W3, b3)
        output = model.forward(x)
        assert (deep_equality(output, expected_output))
    
    def test_cost(self):
        y = np.array([[1, 1, 0]])
        y_hat = np.array([[0.8, 0.9, 0.4]])
        res = CrossEntropy().cost(y, y_hat)
        assert (pytest.approx(0.2797765635793422, res))
    
    def test_linear_backward(self):
        a_prev = np.array([[-0.3224172, -0.38405435, 1.13376944, -1.09989127],
                           [-0.17242821, -0.87785842, 0.04221375, 0.58281521],
                           [-1.10061918, 1.14472371, 0.90159072, 0.50249434],
                           [0.90085595, -0.68372786, -0.12289023, -0.93576943],
                           [-0.26788808, 0.53035547, -0.69166075, -0.39675353]])
        w = np.array([[-0.6871727, -0.84520564, -0.67124613, -0.0126646, -1.11731035],
                      [0.2344157, 1.65980218, 0.74204416, -0.19183555, -0.88762896],
                      [-0.74715829, 1.6924546, 0.05080775, -0.63699565, 0.19091548]])
        b = np.array([[2.10025514],
                      [0.12015895],
                      [0.61720311]])
        d_z = np.array([[1.62434536, -0.61175641, -0.52817175, -1.07296862],
                        [0.86540763, -2.3015387, 1.74481176, -0.7612069],
                        [0.3190391, -0.24937038, 1.46210794, -2.06014071]])
        
        expected_input = np.array([[-1.15171336, 0.06718465, -0.3204696, 2.09812712],
                                   [0.60345879, -3.72508701, 5.81700741, -3.84326836],
                                   [-0.4319552, -1.30987417, 1.72354705, 0.05070578],
                                   [-0.38981415, 0.60811244, -1.25938424, 1.47191593],
                                   [-2.52214926, 2.67882552, -0.67947465, 1.48119548]])
        expected_d_w = np.array([[0.07313866, -0.0976715, -0.87585828, 0.73763362, 0.00785716],
                                 [0.85508818, 0.37530413, -0.59912655, 0.71278189, -0.58931808],
                                 [0.97913304, -0.24376494, -0.08839671, 0.55151192, -0.10290907]])
        expected_b = np.array([[-0.14713786],
                               [-0.11313155],
                               [-0.13209101]])
        
        linear = Linear(5, 3)
        linear._pack_params(w, b)
        linear._cache = a_prev
        
        grads = linear.backward(d_z)
        assert deep_equality(expected_input, grads)
        assert deep_equality((expected_d_w, expected_b), linear._grads.inner())
    
    def test_linear_activation_backward(self):
        d_a_l = np.array([[-0.41675785, -0.05626683]])
        
        a_prev = np.array([[-2.1361961, 1.64027081],
                           [-1.79343559, -0.84174737],
                           [0.50288142, -1.24528809]])
        w = np.array([[-1.05795222, -0.90900761, 0.55145404]])
        b = np.array([[2.29220801]])
        activation_prev = np.array([[0.04153939, -1.11792545]])
        expected_sigmoid_input = np.array([[0.11017994, 0.01105339],
                                           [0.09466817, 0.00949723],
                                           [-0.05743092, -0.00576154]])
        expected_sigmoid_d_w = np.array([[0.10266786, 0.09778551, -0.01968084]])
        expected_sigmoid_d_b = np.array([[-0.05729622]])
        
        expected_relu_input = np.array([[0.44090989, -0.],
                                        [0.37883606, -0.],
                                        [-0.2298228, 0.]])
        expected_relu_d_w = np.array([[ 0.44513824, 0.37371418,-0.10478989]])
        expected_relu_d_b = np.array([[-0.20837892]])
        
        linear_sigmoid = Stacked([Linear(3, 1), Sigmoid()])
        linear_sigmoid._layers[0]._pack_params(w, b)
        linear_sigmoid._layers[0]._cache = a_prev
        linear_sigmoid._layers[1]._cache = activation_prev
        
        grads = linear_sigmoid.backward(d_a_l)
        assert deep_equality(expected_sigmoid_input, grads)
        assert deep_equality((expected_sigmoid_d_w, expected_sigmoid_d_b), linear_sigmoid._layers[0]._grads.inner())
        linear_relu = Stacked([Linear(3, 1), Relu()])
        linear_relu._layers[0]._pack_params(w, b)
        linear_relu._layers[0]._cache = a_prev
        linear_relu._layers[1]._cache = activation_prev
        
        grads = linear_relu.backward(d_a_l)
        assert deep_equality(expected_relu_input, grads)
        assert deep_equality((expected_relu_d_w, expected_relu_d_b), linear_relu._layers[0]._grads.inner())
    
    def test_update_parameter(self):
        W1 = np.array([[-0.41675785, -0.05626683, -2.1361961, 1.64027081],
                       [-1.79343559, -0.84174737, 0.50288142, -1.24528809],
                       [-1.05795222, -0.90900761, 0.55145404, 2.29220801]])
        b1 = np.array([[0.04153939],
                       [-1.11792545],
                       [0.53905832]])
        W2 = np.array([[-0.5961597, -0.0191305, 1.17500122]])
        b2 = np.array([[-0.74787095]])
        dW1 = np.array([[1.78862847, 0.43650985, 0.09649747, -1.8634927],
                        [-0.2773882, -0.35475898, -0.08274148, -0.62700068],
                        [-0.04381817, -0.47721803, -1.31386475, 0.88462238]])
        db1 = np.array([[0.88131804],
                        [1.70957306],
                        [0.05003364]])
        dW2 = np.array([[-0.40467741, -0.54535995, -1.54647732]])
        db2 = np.array([[0.98236743]])

        expected_W1 = np.array([[-0.59562069, -0.09991781, -2.14584584, 1.82662008],
                                [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
                                [-1.0535704, -0.86128581, 0.68284052, 2.20374577]])
        expected_b1 = np.array([[-0.04659241],
                                [-1.28888275],
                                [0.53405496]])
        expected_W2 = np.array([[-0.55569196, 0.0354055, 1.32964895]])
        expected_b2 = np.array([[-0.84610769]])

        model = Stacked(
            [Linear(4, 3),
             Relu(to_cache=True),
             Linear(3, 1),
             Sigmoid(to_cache=True), ],
            to_cache=True)

        model._layers[0]._pack_params(W1, b1)
        model._layers[0]._grads = Grads((dW1, db1), None)
        model._layers[2]._pack_params(W2, b2)
        model._layers[2]._grads = Grads((dW2, db2), None)
        
        learning_rate = 0.1
        model.update_parameters(learning_rate)
        assert deep_equality(model._layers[0].extract_params(), (expected_W1, expected_b1))
        assert deep_equality(model._layers[2].extract_params(), (expected_W2, expected_b2))
