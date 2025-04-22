import unittest

import matrix


def matrix_to_list(string: str) -> list[list[float]]:
    string = string.split('\n')
    if len(string) == 1:
        return [[0 for _ in range(int(string[0].split()[1]))] for _ in range(int(string[0].split()[0]))]

    accuracy = int(string[1])
    return [[round(float(value), accuracy) for value in row.split()] for row in string[2:]]


class TestSparseMatrix(unittest.TestCase):
    def setUp(self):
        self.matrices = [
            # 1
            "3 3\n"
            "2\n"
            "1 0 0\n"
            "0 0 0\n"
            "-7 8.54 0.45",
            # 2
            "4 3\n"
            "2\n"
            "1.43 -234.44 3.4\n"
            "4.43 0 0\n"
            "0 0 0\n"
            "10.21 11.33 12.12",
            # 3
            "5 5\n"
            "2\n"
            "1 0 0 0 0\n"
            "0 7 0 0 0\n"
            "0 0 -3.15 0 0\n"
            "0 0 0 9 0\n"
            "0 0 0 0 25",
            # 4
            "5 5\n"
            "2\n"
            "1 2 3 4 5\n"
            "6 7 8 9 10\n"
            "11 12 13 14 15\n"
            "16 17 18 19 20\n"
            "21 22 23 24 25",
            # 5
            "5 2\n"
            "2\n"
            "1 2\n"
            "6 7\n"
            "11 12\n"
            "16 17\n"
            "21 22",
            # 6
            "2 5\n"
            "2\n"
            "1 2 3 4 5\n"
            "6 7 8 9 10",
            # 7
            "4 4\n"
            "2\n"
            "0 0 0 0\n"
            "0 0 0 0\n"
            "0 0 0 0\n"
            "0 0 0 0",
            # 8
            "4 4",
            # 9
            "3 3\n"
            "2\n"
            "-4 -1 2\n"
            "10 4 -1\n"
            "8 3 1",
            # 10
            "3 3\n"
            "2\n"
            "1 0 0\n"
            "0 0 0\n"
            "-7 8.54 0.45",
            # 11
            "2 2\n"
            "2\n"
            "1 2\n"
            "3 4",
            # 12
            "3 3\n"
            "3\n"
            "5.00002 0 8595\n"
            "0 745.5421 0\n"
            "0 0 43.5431",
            # 13
            "3 3\n"
            "3\n"
            "1.001 2.003 3\n"
            "4 6.424 6\n"
            "7 8 9",
            # 14
            "3 3\n"
            "3\n"
            "0 0 0\n"
            "0 0 0\n"
            "0 0 0",
            # 15
            "3 3\n"
            "3\n"
            "1.001 2.003 3\n"
            "4 6.424 6\n"
            "7 8 9",
            # 16
            "3 3\n"
            "2\n"
            "0 0 0\n"
            "0 0 0\n"
            "0 0 0",
            # 17
            "3 3\n"
            "3\n"
            "1.001 2.003 3\n"
            "4 6.424 6\n"
            "7 8 9",
            # 18
            "3 3\n"
            "2\n"
            "1.001 2.003 3\n"
            "4 6.424 6\n"
            "7 8 9"
        ]
        self.tracers = [
            1.45,  # 1
            None,  # 2
            38.85,  # 3
            65,  # 4
            None,  # 5
            None,  # 6
            0,  # 7
            0,  # 8
            1,  # 9
            1.45,  # 10
            5,  # 11
            794.085,  # 12
            16.425,  # 13
            0,  # 14
            16.425,  # 15
            0,  # 16
            16.425,  # 17
            16.42  # 18
        ]
        self.matrices_sum = [
            None,  # 1 + 2
            None,  # 2 + 3
            "5 5\n"  # 3 + 4
            "2\n"
            "2 2 3 4 5\n"
            "6 14 8 9 10\n"
            "11 12 9.85 14 15\n"
            "16 17 18 28 20\n"
            "21 22 23 24 50\n",
            None,  # 4 + 5
            None,  # 5 + 6
            None,  # 6 + 7
            "4 4\n"  # 7 + 8
            "2\n"
            "0 0 0 0\n"
            "0 0 0 0\n"
            "0 0 0 0\n"
            "0 0 0 0",
            None,  # 8 + 9
            "3 3\n"  # 9 + 10
            "2\n"
            "-3 -1 2\n"
            "10 4 -1\n"
            "1 11.54 1.45",
            None,  # 10 + 11
            None,  # 11 + 12
            "3 3\n"  # 12 + 13
            "3\n"
            "6.00102 2.003 8598\n"
            "4 751.9661 6\n"
            "7 8 52.543",
            "3 3\n"  # 13 + 14
            "3\n"
            "1.001 2.003 3\n"
            "4 6.424 6\n"
            "7 8 9",
            "3 3\n"  # 14 + 15
            "3\n"
            "1.001 2.003 3\n"
            "4 6.424 6\n"
            "7 8 9",
            "3 3\n"  # 15 + 16
            "3\n"
            "1.001 2.003 3\n"
            "4 6.424 6\n"
            "7 8 9",
            "3 3\n"  # 16 + 17
            "2\n"
            "1 2 3\n"
            "4 6.42 6\n"
            "7 8 9",
            "3 3\n"  # 17 + 18
            "3\n"
            "2.001 4.003 6\n"
            "8 12.844 12\n"
            "14 16 18"
        ]
        self.matrices_mul = [
            "3 3\n"  # 1
            "3\n"
            "2 0 0\n"
            "0 0 0\n"
            "-14 17.08 0.9",
            "3 4\n"  # 2
            "2\n"
            "1.43 -234.44 3.4 0\n"
            "4.43 0 0 -1\n"
            "8 9 423.6 -9\n",
            "4 2\n"  # 3    
            "3\n"
            "453.5467 423.6\n"
            "31.01 33.33\n"
            "0 -5.32\n"
            "123 0",
            "2 1\n"  # 4
            "3\n"
            "2.86\n"
            "-468.88",

        ]
        self.matrices_mul_result = [
            "3 4\n"  # 1 * 2
            "3\n"
            "2.86 -468.88 6.8 0\n"
            "0 0 0 0\n"
            "62.844 3290.26 333.64 -25.18",
            "3 2\n"  # 2 * 3
            "2\n"
            "-6621.41 -7226.23\n"
            "1886.21 1876.55\n"
            "2800.47 1435.22",
            "4 1\n"  # 3 * 4
            "3\n"
            "-197320.424\n"
            "-15539.082\n"
            "2494.442\n"
            "351.78"
        ]
        self.determinants = [
            0,  # 1
            None,  # 2
            -4961.25,  # 3
            0,  # 4
            None,  # 5
            None,  # 6
            0,  # 7
            0,  # 8
            -14,  # 9
            0,  # 10
            -2,  # 11
            162315.677,  # 12
            -17.06,  # 13
            0,  # 14
            -17.06,  # 15
            0,  # 16
            -17.06,  # 17
            -17.06  # 18
        ]

    def test_get_element(self):
        for test in range(len(self.matrices)):
            with self.subTest(test=test + 1):
                m = matrix.Matrix(self.matrices[test])
                l = matrix_to_list(self.matrices[test])
                for row in range(1, m.num_rows + 1):
                    for column in range(1, m.num_columns + 1):
                        self.assertEqual(l[row - 1][column - 1], m[row, column])

    def test_tracer(self):
        for test in range(len(self.matrices)):
            with self.subTest(test=test):
                m = matrix.Matrix(self.matrices[test])
                if self.tracers[test] is None:
                    with self.assertRaises(Exception):
                        _ = m.get_tracer()
                else:
                    self.assertEqual(self.tracers[test], m.get_tracer())

    def test_transpose(self):
        for test in range(len(self.matrices)):
            with self.subTest(test=test + 1):
                m = matrix.Matrix(self.matrices[test])
                l = matrix_to_list(self.matrices[test])
                transposed = [[l[j][i] for j in range(len(l))] for i in range(len(l[0]))]
                self.assertEqual(m.transpose().get_list(), transposed)

    def test_matrices_sum(self):
        for test in range(len(self.matrices) - 1):
            with self.subTest(test=test + 1):
                m1 = matrix.Matrix(self.matrices[test])
                m2 = matrix.Matrix(self.matrices[test + 1])
                if self.matrices_sum[test] is None:
                    with self.assertRaises(Exception):
                        _ = m1 + m2
                else:
                    self.assertEqual(matrix.Matrix(self.matrices_sum[test]).get_list(), (m1 + m2).get_list())

    def test_matrices_mul(self):
        for test in range(len(self.matrices_mul) - 1):
            with self.subTest(test=test + 1):
                m1 = matrix.Matrix(self.matrices_mul[test])
                m2 = matrix.Matrix(self.matrices_mul[test + 1])
                self.assertEqual(matrix.Matrix(self.matrices_mul_result[test]).get_list(), (m1 * m2).get_list())

    def test_determinant(self):
        for test in range(len(self.matrices)):
            with self.subTest(test=test + 1):
                m = matrix.Matrix(self.matrices[test])
                if self.determinants[test] is None:
                    with self.assertRaises(Exception):
                        m.get_determinant()
                else:
                    self.assertAlmostEqual(self.determinants[test], m.get_determinant(), 0)
