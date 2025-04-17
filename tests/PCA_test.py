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
            "9 4\n"
            "2\n"
            "2 4 54.5 3.5\n"
            "23 43 45 56\n"
            "34 45 56 67\n"
            "213 94.5 35 34\n"
            "39 24 59 34\n"
            "23 45 56 67\n"
            "45 95 95 45 \n"
            "777 33 43.2 45.4\n"
            "2.1 2.3 6.5 4.5\n"
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
            "9 4\n"
            "2\n"
            "-126.68 -38.87 4.48 -36.10\n"
            "-105.68 0.13 -5.02 16.40\n"
            "-94.68 2.13 5.98 27.40\n"
            "84.32 51.63 -15.02 -5.60\n"
            "-89.68 -18.87 8.98 -5.60\n"
            "-105.68 2.13 5.98 27.40\n"
            "-83.68 52.13 44.98 5.40\n"
            "648.32 -9.87 -6.82 5.80\n"
            "-126.58 -40.57 -43.52 -35.10\n"
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
            "4 4\n"
            "2\n"
            "63230.58 612.93 -747.96 641.44\n"
            "612.93 1125.32 385.43 373.36\n"
            "-747.96 385.43 548.34 231.04\n"
            "641.44 373.36 231.04 553.90"
        ]
        self.eigenvalues = [
            # 1
            [-0.000000,
             0.140956,
             43.237078],
            # 2
            [14219.241125,
             2.254053,
             47.022806],
            # 3
            [0.000000,
             1.272515,
             7.878683,
             17.160720,
             126.872581],
            # 4
            [1513.904082,
             302.346435,
             389.856198,
             63252.044674]
        ]

    def test_center_data(self):
        for test in range(len(self.centered_data)):
            with self.subTest(test=test + 1):
                m = Matrix(self.matrices[test])
                centered_m = PCA.center_data(m)
                self.assertEqual(centered_m.get_list(), Matrix(self.centered_data[test]).get_list())

    def test_covariance_matrix(self):
        for test in range(len(self.covariance_data)):
            with self.subTest(test=test + 1):
                m = Matrix(self.matrices[test])
                centered_m = PCA.center_data(m)
                cov_matrix = PCA.covariance_matrix(centered_m)
                correct_matrix = Matrix(self.covariance_data[test])
                for row in range(1, m.num_columns + 1):
                    for column in range(1, m.num_columns + 1):
                        self.assertAlmostEqual(cov_matrix[row, column], correct_matrix[row, column], 2)

    def test_eigenvalues(self):
        for test in range(len(self.eigenvalues)):
            with self.subTest(test=test + 1):
                if test == 4:
                    pass
                m = Matrix(self.matrices[test])
                eigenvalues = PCA.get_eigenvalues(m)
                self.assertEqual(eigenvalues, sorted(self.eigenvalues[test]))


if __name__ == '__main__':
    unittest.main()
