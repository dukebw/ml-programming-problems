# Copyright 2018 Brendan Duke.
#
# This file is part of ML Programming Problems.
#
# ML Programming Problems is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# ML Programming Problems is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# ML Programming Problems. If not, see <http://www.gnu.org/licenses/>.

from autograd import grad
import autograd.numpy as np


def _rosenbrock(x):
    assert len(x) == 2

    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2


def _get_loss(weights, x_in, y_gt):
    w_h, b_h, w_out, b_out = weights

    h_0 = np.matmul(w_h, x_in)
    h_0 += b_h

    y_out = np.matmul(w_out, h_0)
    y_out += b_out

    loss = 0.5*(y_out - y_gt)**2

    return loss


def test_mlp():
    D = [2, 8, 1]

    x_in = np.random.randn(D[0])
    # Write the forward and backward pass for an MLP with one hidden layer.
    w_h = np.random.randn(D[0]*D[1]).reshape([D[1], D[0]])
    b_h = np.random.randn(D[1])

    w_out = np.random.randn(D[1]*D[2]).reshape([D[2], D[1]])
    b_out = np.random.randn(D[2])

    # Forward.
    h_0 = np.matmul(w_h, x_in)
    h_0 += b_h

    y_out = np.matmul(w_out, h_0)
    y_out += b_out

    y_gt = _rosenbrock(x_in)
    y_bar = y_out - y_gt
    loss = 0.5*y_bar**2

    grad_b_out = y_bar
    y_bar = np.expand_dims(y_bar, -1)
    h_0 = np.expand_dims(h_0, -1)
    grad_w_out = np.matmul(y_bar, h_0.transpose())

    h_0_bar = np.matmul(w_out.transpose(), y_bar)

    grad_b_h = h_0_bar
    h_0_bar = np.expand_dims(h_0_bar, -1)
    x_in = np.expand_dims(x_in, -1)
    grad_w_h = np.matmul(h_0_bar, x_in.transpose())

    loss_grad_fn = grad(_get_loss)
    autograd_grad = loss_grad_fn([w_h, b_h, w_out, b_out], x_in.squeeze(), y_gt)

    eps = np.finfo(np.float32).eps
    assert np.max(np.abs(autograd_grad[0].flatten() - grad_w_h.flatten())) < eps
    assert np.max(np.abs(autograd_grad[1].flatten() - grad_b_h.flatten())) < eps
    assert np.max(np.abs(autograd_grad[2].flatten() - grad_w_out.flatten())) < eps
    assert np.max(np.abs(autograd_grad[3].flatten() - grad_b_out.flatten())) < eps

    # softmax_i = e^y_i / \sum_j e^y_j
    # sigmoid: 1/(1 + e^-x)


if __name__ == '__main__':
    test_mlp()
