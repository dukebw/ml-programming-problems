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


def strstr(haystack, needle):
    kmp_table = np.zeros(len(needle), dtype=np.int32)
    kmp_table[0] = -1
    kmp_table[1] = 1

    #  P A R T I C I P A T E   I N   P A R A C H U T E
    # -1 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 1 2 3 0 0 0 0 0
    n_i = 0
    h_i = 0
    while (h_i + n_i) < len(needle):
        kmp_table[h_i + n_i] = None


def test_strstr():
    test_cases = [(('hello', 'll'), 2),
                  (('aaaaa', 'bba'), -1)]
    for case in test_cases:
        assert strstr(*case[0]) == case[1]


if __name__ == '__main__':
    test_strstr()
