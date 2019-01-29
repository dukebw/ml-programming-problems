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


def test_binary_search():
    array = np.empty(1024)

    for _ in range(128):
        array[0] = np.random.randint(10)
        for i in range(1, len(array)):
            array[i] = array[i - 1] + np.random.randint(1, 10)

        search_i = np.random.randint(len(array))
        to_search = array[search_i]

        # Binary search algorithm here.

        assert mid_i == search_i


if __name__ == '__main__':
    test_binary_search()
