# Created by @nyutal on 06/05/2020
import numpy as np
import pytest
from course_1.week_3.nn_model import NNModel


@pytest.fixture()
def model():
    return NNModel(4)


@pytest.fixture()
def dummy_data():
    np.random.seed(1)
    x = np.random.randn(2, 1)
    y = np.random.randint(0, 1, size=1)
    return x, y


class TestWeek3:
    def test_init_params(self, model: NNModel, dummy_data):
        expected_w_1 = np.array(
            [[-0.00416758, -0.00056267], [-0.02136196, 0.01640271], [-0.01793436, -0.00841747],
             [0.00502881, -0.01245288]])
        expected_b_1 = np.array([[0.],
                                 [0.],
                                 [0.],
                                 [0.]])
        expected_w_2 = np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]])
        expected_b_2 = np.array([[0.]])
        
        np.random.seed(2)
        x, y = dummy_data
        w_1, b_1, w_2, b_2 = model._init_params(x, y)
        assert np.allclose(w_1, expected_w_1)
        assert np.allclose(b_1, expected_b_1)
        assert np.allclose(w_2, expected_w_2)
        assert np.allclose(b_2, expected_b_2)
    
    def test_forward_propagation(self, model: NNModel):
        x = np.array([[1.62434536, -0.61175641, -0.52817175],
                      [-1.07296862, 0.86540763, -2.3015387]])
        parameters = (
            np.array([[-0.00416758, -0.00056267],
                      [-0.02136196, 0.01640271],
                      [-0.01793436, -0.00841747],
                      [0.00502881, -0.01245288]]),
            np.array([[1.74481176],
                      [-0.7612069],
                      [0.3190391],
                      [-0.24937038]]),
            np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
            np.array([[-1.3]]))
        
        expected_z_1 = np.array(
            [[1.7386459, 1.74687437, 1.74830797],
             [-0.81350569, -0.73394355, -0.78767559],
             [0.29893918, 0.32272601, 0.34788465],
             [-0.2278403, -0.2632236, -0.22336567]])
        expected_a_1 = np.array(
            [[0.9400694, 0.94101876, 0.94118266],
             [-0.67151964, -0.62547205, -0.65709025],
             [0.29034152, 0.31196971, 0.33449821],
             [-0.22397799, -0.25730819, -0.2197236]])
        expected_z_2 = np.array([[-1.30737426, -1.30844761, -1.30717618]])
        expected_a_2 = np.array([[0.21292656, 0.21274673, 0.21295976]])
        z_1, a_1, z_2, a_2 = model._forward_propagation(parameters, x)
        assert np.allclose(z_1, expected_z_1)
        assert np.allclose(a_1, expected_a_1)
        assert np.allclose(z_2, expected_z_2)
        assert np.allclose(a_2, expected_a_2)
    
    def test_compute_cost(self, model: NNModel):
        a_2 = np.array([[0.5002307, 0.49985831, 0.50023963]])
        y = np.array([[True, False, False]])
        expected_cost = 0.6930587610394646
        cost = model._compute_cost(a_2, y)
        assert cost == expected_cost
    
    def test_backward_propagation(self, model: NNModel):
        parameters = (
            np.array(
                [[-0.00416758, -0.00056267],
                 [-0.02136196, 0.01640271],
                 [-0.01793436, -0.00841747],
                 [0.00502881, -0.01245288]]),
            np.array([[0.],
                      [0.],
                      [0.],
                      [0.]]),
            np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
            np.array([[0.]]))
        a_1 = np.array([[-0.00616578, 0.0020626, 0.00349619],
                        [-0.05225116, 0.02725659, -0.02646251],
                        [-0.02009721, 0.0036869, 0.02883756],
                        [0.02152675, -0.01385234, 0.02599885]])
        a_2 = np.array([[0.5002307, 0.49985831, 0.50023963]])
        x = np.array([[1.62434536, -0.61175641, -0.52817175],
                      [-1.07296862, 0.86540763, -2.3015387]])
        y = np.array([[True, False, True]])
        
        expected_dw_1 = np.array([[0.00301023, -0.00747267],
                                  [0.00257968, -0.00641288],
                                  [-0.00156892, 0.003893],
                                  [-0.00652037, 0.01618243]])
        expected_db_1 = np.array([[0.00176201],
                                  [0.00150995],
                                  [-0.00091736],
                                  [-0.00381422]])
        expected_dw_2 = np.array([[0.00078841, 0.01765429, -0.00084166, -0.01022527]])
        expected_db_2 = np.array([[-0.16655712]])
        
        dw_1, db_1, dw_2, db_2 = model._backward_propagation(parameters, a_1, a_2, x, y)
        assert np.allclose(dw_1, expected_dw_1)
        assert np.allclose(db_1, expected_db_1)
        assert np.allclose(dw_2, expected_dw_2)
        assert np.allclose(db_2, expected_db_2)
    
    def test_update_parameters(self, model: NNModel):
        parameters = (
            np.array(
                [[-0.00615039, 0.0169021],
                 [-0.02311792, 0.03137121],
                 [-0.0169217, -0.01752545],
                 [0.00935436, -0.05018221]]),
            np.array([[-0.0000009],
                      [0.00000816],
                      [0.0000006],
                      [-0.00000255]]),
            np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
            np.array([[0.0000915]]))
        grads = (
            np.array([[0.00023322, -0.00205423],
                      [0.00082222, -0.00700776],
                      [-0.00031831, 0.0028636],
                      [-0.00092857, 0.00809933]]),
            np.array([[0.00000011],
                      [-0.00000382],
                      [-0.00000019],
                      [0.00000055]]),
            np.array([[-0.00001757, 0.00370231, -0.00125683, -0.00255715]]),
            np.array([[-0.00001089]]))
        expected_w_1 = np.array(
            [[-0.00643025, 0.01936718],
             [-0.02410458, 0.03978052],
             [-0.01653973, -0.02096177],
             [0.01046864, -0.05990141]])
        expected_b_1 = np.array(
            [[-0.00000102],
             [0.00001274],
             [0.00000083],
             [-0.0000032]])
        expected_w_2 = np.array([[-0.01041081, -0.04463285, 0.01758031, 0.04747113]])
        expected_b_2 = np.array([[0.00010457]])
        w_1, b_1, w_2, b_2 = model._update_parameters(parameters, grads, 1.2)
        assert np.allclose(w_1, expected_w_1)
        assert np.allclose(b_1, expected_b_1, atol=1.e-4, )
        assert np.allclose(w_2, expected_w_2)
        assert np.allclose(b_2, expected_b_2)
    
    def test_fit(self, model):
        x = np.array([[1.62434536, -0.61175641, -0.52817175],
                      [-1.07296862, 0.86540763, -2.3015387]])
        y = np.array([[True, False, True]])
        
        expected_cost_first_iteration = 0.692739
        results = model.fit(x, y, num_iteration=1, learning_rate=1.2, seed=3, verbose=False)
        assert pytest.approx(expected_cost_first_iteration, results['cost'][0])
    
    def test_predict(self, model: NNModel):
        x = np.array([[1.62434536, -0.61175641, -0.52817175],
                      [-1.07296862, 0.86540763, -2.3015387]])
        w_1 = np.array(
            [[-0.00615039, 0.0169021],
             [-0.02311792, 0.03137121],
             [-0.0169217, -0.01752545],
             [0.00935436, -0.05018221]])
        w_2 = np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]])
        b_1 = np.array([[-0.0000009],
                        [0.00000816],
                        [0.0000006],
                        [-0.00000255]])
        b_2 = np.array([[0.0000915]])
        expected_mean = 0.666666666667
        
        model._save_parameters(w_1, b_1, w_2, b_2)
        res = model.predict(x)
        assert pytest.approx(expected_mean, res.mean())
