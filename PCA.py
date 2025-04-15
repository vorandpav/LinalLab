import math
from copy import deepcopy
from typing import List

from matrix import Matrix


class PCA():
    def center_data(X: 'Matrix') -> 'Matrix':
        """
        Центрирование данных в матрице X.

        :param X: Матрица, которую нужно центрировать.
        :return: Центрированная матрица.
        """
        columns_sum = [0 for _ in range(X.num_columns)]
        for row in range(1, X.num_rows + 1):
            for element in range(X.row_sizes[row - 1], X.row_sizes[row]):
                columns_sum[X.column_indices[element]] \
                    += X.values[element]
        columns_mean = [sum / X.num_rows for sum in columns_sum]

        mean_X = Matrix(f"{X.num_rows} {X.num_columns}\n"
                        f"{X.accuracy}\n"
                        f"{' '.join([' '.join(str(mean) for mean in columns_mean) + '\n' for _ in range(X.num_rows)])}")
        X_centered = X - mean_X
        return X_centered

    def covariance_matrix(X_centered: 'Matrix') -> 'Matrix':
        """
        Вычисление ковариационной матрицы для центрированной матрицы X.

        :param X_centered: Центрированная матрица.
        :return: Ковариационная матрица.
        """
        X_centered_transposed = X_centered.transpose()
        X_centered_transposed.accuracy = X_centered.accuracy * 2
        covariance_matrix = X_centered_transposed * X_centered
        covariance_matrix.accuracy = X_centered.accuracy
        covariance_matrix /= X_centered.num_rows - 1
        return covariance_matrix

    def _characteristic_polynomial_value(covariance_matrix: 'Matrix', value: float) -> float:
        """
        Вычисление значения характеристического многочлена для заданного значения.
        :param covariance_matrix: Ковариационная матрица.
        :param value: Значение, для которого вычисляется характеристический многочлен.
        :return: Значение характеристического многочлена.
        """
        matrix = deepcopy(covariance_matrix)
        for row in range(1, matrix.num_rows + 1):
            matrix[row, row] -= value
        return matrix.get_determinant()

    def get_eigenvalues(X: 'Matrix', tolerance: float = 1e-6) -> List[float]:
        """
        Вычисление собственных значений ковариационной матрицы для заданной матрицы X.

        :param X: Матрица, для которой вычисляются собственные значения.
        :param tolerance: Допустимая погрешность для вычисления собственных значений.
        :return: Собственные значения.
        """
        matrix = deepcopy(X)
        matrix.accuracy = 100
        covariance_matrix = PCA.covariance_matrix(PCA.center_data(matrix))
        eigenvalues = PCA.find_eigenvalues(covariance_matrix, tolerance)
        return eigenvalues

    def find_eigenvalues(covariance_matrix: 'Matrix', tolerance: float = 1e-6) -> List[float]:
        """
        Вычисление собственных значений ковариационной матрицы.

        :param covariance_matrix: Ковариационная матрица.
        :param tolerance: Допустимая погрешность для вычисления собственных значений.
        :return: Собственные значения.

        Осторожно, при низкой точности нахождения ковариационной матрицы, ответ не находится из-за изначальных погрешностей.
        """
        eigenvalues = []
        matrix = deepcopy(covariance_matrix)
        matrix.accuracy = 100
        zero_eigenvalue = abs(matrix.get_determinant()) < tolerance

        num_intervals = matrix.num_rows

        while len(eigenvalues) != matrix.num_rows - zero_eigenvalue:
            for interval_index in range(len(eigenvalues) + 1):
                if len(eigenvalues) == 0:
                    lower_bound = tolerance
                    higher_bound = matrix.get_tracer() + tolerance
                elif interval_index == 0:
                    lower_bound = tolerance
                    higher_bound = eigenvalues[0] - tolerance
                elif interval_index == len(eigenvalues):
                    lower_bound = eigenvalues[-1] + tolerance
                    higher_bound = matrix.get_tracer() + tolerance
                else:
                    lower_bound = eigenvalues[interval_index - 1] + tolerance
                    higher_bound = eigenvalues[interval_index] - tolerance
                if higher_bound - lower_bound < tolerance:
                    continue
                interval_length = (higher_bound - lower_bound) / num_intervals
                for interval in range(num_intervals):
                    if len(eigenvalues) == matrix.num_rows - zero_eigenvalue:
                        break
                    low = lower_bound + interval * interval_length
                    high = lower_bound + (interval + 1) * interval_length
                    characteristic_value_low = PCA._characteristic_polynomial_value(matrix, low)
                    characteristic_value_high = PCA._characteristic_polynomial_value(matrix, high)
                    if characteristic_value_low * characteristic_value_high < 0:
                        if characteristic_value_low > 0:
                            low, high = high, low
                        eigenvalue = (low + high) / 2
                        while True:
                            characteristic_value = PCA._characteristic_polynomial_value(matrix, eigenvalue)
                            if abs(high - low) < tolerance:
                                eigenvalues.append(round(eigenvalue, int(math.log10(1 / tolerance))))
                                break
                            if characteristic_value > 0:
                                high = eigenvalue
                            else:
                                low = eigenvalue
                            eigenvalue = (low + high) / 2
            num_intervals *= 10
        if zero_eigenvalue:
            eigenvalues.append(0)

        return sorted(eigenvalues)


if __name__ == '__main__':
    m = Matrix("4 3\n"
               "2\n"
               "1.43 -234.44 3.4\n"
               "4.43 0 0\n"
               "0 0 0\n"
               "10.21 11.33 12.12")
    centred_m = PCA.covariance_matrix(PCA.center_data(m))
    [print(i) for i in centred_m.get_list()]
    eigenvalues = PCA.get_eigenvalues(m, 1e-6)
    print([f"{i}" for i in eigenvalues])
