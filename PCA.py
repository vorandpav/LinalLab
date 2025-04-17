import math
from copy import deepcopy
from typing import List
from matrix import Matrix


class PCA:
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
        covariance_matrix = X_centered_transposed * X_centered
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

    def find_eigenvalues(covariance_matrix: 'Matrix', tolerance: float = 1e-6,
                         intial_num_intervals=10, max_num_intervals=1000) -> List[float]:
        """
        Вычисление собственных значений ковариационной матрицы.

        :param covariance_matrix: Ковариационная матрица.
        :param tolerance: Допустимая погрешность для вычисления собственных значений.
        :param intial_num_intervals: Изначальное число интервалов для поиска на разбиении.
        :param max_num_intervals: Максимальное число интервалов для поиска на разбиении.
        :return: Собственные значения.

        Осторожно, при низкой точности нахождения ковариационной матрицы, ответ не находится из-за изначальных погрешностей.
        """
        eigenvalues = []
        matrix = deepcopy(covariance_matrix)
        zero_eigenvalue = abs(matrix.get_determinant()) < tolerance

        one_eigenvalue = abs(PCA._characteristic_polynomial_value(matrix, 1)) < tolerance
        eigenvalues.append(1)

        num_intervals = intial_num_intervals

        while len(eigenvalues) - 1 != matrix.num_rows - zero_eigenvalue and num_intervals <= max_num_intervals:
            for interval_index in range(len(eigenvalues) + 1):
                if interval_index == 0:
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
                current_num_intervals = num_intervals
                if interval_length < tolerance:
                    current_num_intervals = int((higher_bound - lower_bound) / tolerance) + 1
                    interval_length = (higher_bound - lower_bound) / num_intervals

                for interval in range(current_num_intervals):
                    if len(eigenvalues) - 1 == matrix.num_rows - zero_eigenvalue:
                        break
                    low = lower_bound + interval * interval_length
                    high = lower_bound + (interval + 1) * interval_length
                    characteristic_value_low = PCA._characteristic_polynomial_value(matrix, low)
                    characteristic_value_high = PCA._characteristic_polynomial_value(matrix, high)
                    if characteristic_value_low == 0:
                        eigenvalues.append(round(low, int(math.log10(1 / tolerance))))
                        continue
                    if characteristic_value_high == 0:
                        eigenvalues.append(round(high, int(math.log10(1 / tolerance))))
                        continue
                    if characteristic_value_low * characteristic_value_high < 0:
                        if characteristic_value_low > 0:
                            low, high = high, low
                        eigenvalue = (low + high) / 2
                        while True:
                            characteristic_value = PCA._characteristic_polynomial_value(matrix, eigenvalue)
                            if abs(high - low) < tolerance / 10:
                                eigenvalues.append(round(eigenvalue, int(math.log10(1 / tolerance))))
                                eigenvalues.sort()
                                break
                            if characteristic_value > 0:
                                high = eigenvalue
                            else:
                                low = eigenvalue
                            eigenvalue = (low + high) / 2
            num_intervals *= 10
        if zero_eigenvalue:
            eigenvalues.append(0)
        if not one_eigenvalue:
            eigenvalues = [i for i in eigenvalues if i != 1]

        return sorted(eigenvalues)

    def find_eigenvectors(C: 'Matrix', eigenvalues: List[float]) -> List['Matrix']:
        """
        Вход:
        C: матрица ковариаций (m×m)
        eigenvalues: список собственных значений
        Выход: список собственных векторов (каждый вектор - объект Matrix)
        """
        pass

    def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
        """
        Вычисление доли абсолютной дисперсии для первых k собственных значений.

        :param eigenvalues: Собственные значения.
        :param k: Количество собственных значений.
        :return: Доля абсолютной дисперсии.
        """

        return sum(eigenvalues[-k:]) / sum(eigenvalues)


if __name__ == '__main__':
    m = Matrix(
        "10 10\n"
        "10\n"
        "5.61 4.95 -3.93 -3.22 -4.53 -2.4 9.01 -4.14 -3.95 6.05\n"
        "0.35 -2.57 3.13 -3.83 4.72 9.77 1.28 9.89 9.98 4.85\n"
        "8.26 4.7 7.32 6.76 -3.62 1.92 4.76 -1.45 1.95 -0.4\n"
        "9.82 -4.97 9.3 6.59 2.17 -1.2 3.61 4.36 -2.86 5.58\n"
        "8.51 -1.64 -1.21 6.8 -1.76 6.77 8.68 -0.79 -1.38 4.28\n"
        "-2.71 9.41 7.4 3.09 -2.33 9.1 -2.98 4.91 4.91 -0.18\n"
        "0.96 3.94 6.37 8.43 -4.03 8.45 -3.16 9.88 -2.89 -1.22\n"
        "4.45 6.4 1.75 1.01 6.02 9.72 4.17 0.05 9.02 0.87\n"
        "7.42 -1.41 -2.78 -1.57 -0.75 -4.18 -4.46 1.39 6.18 1.77\n"
        "0.91 -1.52 -4.95 2.28 1.13 5.71 2.04 4.32 -0.94 5.66\n")
    eigenvalues = PCA.get_eigenvalues(m, 1e-6)
    print([f"{i}" for i in eigenvalues])
    print([PCA.explained_variance_ratio(eigenvalues, i + 1) for i in range(len(eigenvalues))])
