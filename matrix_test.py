import unittest
import matrix


def matrix_to_list(string: str) -> list[list[float]]:
    return [[float(value) for value in row.split()] for row in string.split('\n')[1:]]


class TestSparseMatrix(unittest.TestCase):
    matrixes = [
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
        "6 7 8 9 10"
    ]

    tracers = [
        1.45435,
        None,
        38.85,
        65,
        None,
        None
    ]

    def test_get_element(self):
        for test in range(len(self.matrixes)):
            m = matrix.Matrix(self.matrixes[test])
            l = matrix_to_list(self.matrixes[test])
            for row in range(1, m.num_rows + 1):
                for column in range(1, m.num_columns + 1):
                    self.assertEqual(m[row, column], l[row - 1][column - 1])

    def test_tracer(self):
        for test in range(len(self.matrixes)):
            m = matrix.Matrix(self.matrixes[test])
            if self.tracers[test] is None:
                self.assertRaises(Exception, m.get_tracer)
            else:
                self.assertEqual(m.get_tracer(), self.tracers[test])


if __name__ == "__main__":
    unittest.main()
