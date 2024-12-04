import unittest

import matrix


def matrix_to_list(string: str) -> list[list[float]]:
    string = string.split('\n')
    if len(string) == 1:
        return [[0 for _ in range(int(string[0].split()[1]))] for _ in range(int(string[0].split()[0]))]

    return [[float(value) for value in row.split()] for row in string[1:]]


class TestSparseMatrix(unittest.TestCase):
    matrices = [
        "3 3\n"
        "1 0 0\n"
        "0 0 0\n"
        "-7 8.543 0.45435",

        "4 3\n"
        "1.43 -234.44 3.4\n"
        "4.432 0 0\n"
        "0 0 0\n"
        "10.213 11.33 12.123",

        "5 5\n"
        "1 0 0 0 0\n"
        "0 7 0 0 0\n"
        "0 0 -3.15 0 0\n"
        "0 0 0 9 0\n"
        "0 0 0 0 25",

        "5 5\n"
        "1 2 3 4 5\n"
        "6 7 8 9 10\n"
        "11 12 13 14 15\n"
        "16 17 18 19 20\n"
        "21 22 23 24 25",

        "5 2\n"
        "1 2\n"
        "6 7\n"
        "11 12\n"
        "16 17\n"
        "21 22",

        "2 5\n"
        "1 2 3 4 5\n"
        "6 7 8 9 10",

        "4 4\n"
        "0 0 0 0\n"
        "0 0 0 0\n"
        "0 0 0 0\n"
        "0 0 0 0",

        "4 4",

        "3 3\n"
        "-4 -1 2\n"
        "10 4 -1\n"
        "8 3 1"
    ]

    def test_get_element(self):
        for test in range(len(self.matrices)):
            m = matrix.Matrix(self.matrices[test])
            l = matrix_to_list(self.matrices[test])
            for row in range(1, m.num_rows + 1):
                for column in range(1, m.num_columns + 1):
                    self.assertEqual(l[row - 1][column - 1], m[row, column])

    tracers = [
        1.45435,
        None,
        38.85,
        65,
        None,
        None,
        0,
        0,
        1
    ]

    def test_tracer(self):
        for test in range(len(self.matrices)):
            m = matrix.Matrix(self.matrices[test])
            if self.tracers[test] is None:
                with self.assertRaises(Exception):
                    _ = m.get_tracer()
            else:
                self.assertEqual(self.tracers[test], m.get_tracer())

    matrices_sum = [
        None,
        None,
        "5 5\n"
        "2 2 3 4 5\n"
        "6 14 8 9 10\n"
        "11 12 9.85 14 15\n"
        "16 17 18 28 20\n"
        "21 22 23 24 50\n",
        None,
        None,
        None,
        "4 4\n"
        "0 0 0 0\n"
        "0 0 0 0\n"
        "0 0 0 0\n"
        "0 0 0 0",
        None
    ]

    def test_matrices_sum(self):
        for test in range(len(self.matrices) - 1):
            m1 = matrix.Matrix(self.matrices[test])
            m2 = matrix.Matrix(self.matrices[test + 1])
            if self.matrices_sum[test] is None:
                with self.assertRaises(Exception):
                    _ = m1 + m2
            else:
                self.assertEqual(matrix.Matrix(self.matrices_sum[test]).get_list(), (m1 + m2).get_list())

    determinants = [
        0,
        None,
        -4961.25,
        0,
        None,
        None,
        0,
        0,
        -14
    ]

    def test_determinant(self):
        for test in range(len(self.matrices)):
            m = matrix.Matrix(self.matrices[test])
            if self.determinants[test] is None:
                with self.assertRaises(Exception):
                    m.get_determinant()
            else:
                self.assertAlmostEqual(self.determinants[test], m.get_determinant(), 5)
