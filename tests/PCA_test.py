import unittest
from PCA import PCA
from matrix import Matrix


class MyTestCase(unittest.TestCase):
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
            "21 22 23 24 25"
        ]
        self.centered_data = [
            # 1
            "3 3\n"
            "2\n"
            "3 -2.85 -0.15\n"
            "2 -2.85 -0.15\n"
            "-5 5.69 0.3",
            # 2
            "4 3\n"
            "2\n"
            "-2.59 -178.66 -0.48\n"
            "0.41 55.78 -3.88\n"
            "-4.02 55.78 -3.88\n"
            "6.19 67.11 8.24",
            # 3
            "5 5\n"
            "2\n"
            "0.8 -1.4 0.63 -1.8 -5.0\n"
            "-0.2 5.6 0.63 -1.8 -5.0\n"
            "-0.2 -1.4 -2.52 -1.8 -5.0\n"
            "-0.2 -1.4 0.63 7.2 -5.0\n"
            "-0.2 -1.4 0.63 -1.8 20.0",
            # 4
            "5 5\n"
            "2\n"
            "-10 -10 -10 -10 -10\n"
            "-5 -5 -5 -5 -5\n"
            "0 0 0 0 0\n"
            "5 5 5 5 5\n"
            "10 10 10 10 10"
        ]
        self.covariance_data = [
            # 1
            "3 3\n"
            "2\n"
            "19.00 -21.35 -1.12\n"
            "-21.35 24.31 1.28\n"
            "-1.12 1.28 0.07",
            # 2
            "3 3\n"
            "2\n"
            "20.45 225.59 22.09\n"
            "225.59 14215.32 68.63\n"
            "22.09 68.63 32.75",
            # 3
            "5 5\n"
            "2\n"
            "0.2 -0.35 0.16 -0.45 -1.25\n"
            "-0.35 9.8 1.1 -3.15 -8.75\n"
            "0.16 1.1 1.98 1.42 3.94\n"
            "-0.45 -3.15 1.42 16.2 -11.25\n"
            "-1.25 -8.75 3.94 -11.25 125.0\n",
            # 4
            "5 5\n"
            "2\n"
            "62.50 62.50 62.50 62.50 62.50\n"
            "62.50 62.50 62.50 62.50 62.50\n"
            "62.50 62.50 62.50 62.50 62.50\n"
            "62.50 62.50 62.50 62.50 62.50\n"
            "62.50 62.50 62.50 62.50 62.50"
        ]

    def test_center_data(self):
        for test in range(len(self.centered_data)):
            with self.subTest(test=test + 1):
                m = Matrix(self.matrices[test])
                centered_m = PCA.center_data(m)
                self.assertEqual(centered_m.get_list(), Matrix(self.centered_data[test]).get_list())

    def test_covariance_matrix(self):
        for test in range(len(self.centered_data)):
            with self.subTest(test=test + 1):
                m = Matrix(self.matrices[test])
                centered_m = PCA.center_data(m)
                cov_matrix = PCA.covariance_matrix(centered_m)
                correct_matrix = Matrix(self.covariance_data[test])
                for row in range(1, m.num_columns + 1):
                    for column in range(1, m.num_columns + 1):
                        self.assertAlmostEqual(cov_matrix[row, column], correct_matrix[row, column], 2)


if __name__ == '__main__':
    unittest.main()
