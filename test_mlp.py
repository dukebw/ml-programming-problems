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

import numpy as np


def test_mlp():
    D = [16, 16, 1]

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

    # softmax_i = e^y_i / \sum_j e^y_j
    # sigmoid: 1/(1 + e^-x)


if __name__ == '__main__':
    test_mlp()
