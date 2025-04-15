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

    def characteristic_polynomial_value(covariance_matrix: 'Matrix', value: float) -> float:
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

    def find_eigenvalues(covariance_matrix: 'Matrix', tolerance: float = 1e-6) -> List[float]:
        """
        Вычисление собственных значений ковариационной матрицы.

        :param covariance_matrix: Ковариационная матрица.
        :param tolerance: Допустимая погрешность для вычисления собственных значений.
        :return: Собственные значения.
        """
        eigenvalues = []
        lower_bound = 0
        higher_bound = covariance_matrix.get_tracer()
        num_intervals = covariance_matrix.num_rows

        while len(eigenvalues) != covariance_matrix.num_rows:
            eigenvalues = []
            interval_length = (higher_bound - lower_bound) / num_intervals
            for interval in range(num_intervals):
                low = lower_bound + interval * interval_length
                high = lower_bound + (interval + 1) * interval_length
                characteristic_value_low = PCA.characteristic_polynomial_value(covariance_matrix, low)
                characteristic_value_high = PCA.characteristic_polynomial_value(covariance_matrix, high)
                if characteristic_value_low * characteristic_value_high < 0:
                    eigenvalue = (low + high) / 2
                    while True:
                        characteristic_value = PCA.characteristic_polynomial_value(covariance_matrix, eigenvalue)
                        if high - low < tolerance:
                            eigenvalues.append(eigenvalue)
                            break
                        if characteristic_value > 0:
                            high = eigenvalue
                        else:
                            low = eigenvalue
                        eigenvalue = (low + high) / 2
            num_intervals *= 10
        return sorted(eigenvalues)


if __name__ == '__main__':
    m = Matrix("3 3\n"
               "10000\n"
               "5.00002 0 8595\n"
               "0 745.5421 0\n"
               "0 0 43.5431", )
    centred_m = PCA.covariance_matrix(PCA.center_data(m))
    [print(i) for i in centred_m.get_list()]
    eigenvalues = PCA.find_eigenvalues(centred_m, 0.000001)
    print([f"{i:.6f}" for i in eigenvalues])
