import math
import random
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from matrix import Matrix
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class PCA:
    def mean_vector(X: 'Matrix') -> 'Matrix':
        """
        Вычисление вектора средних значений для каждой колонки матрицы X.

        :param X: Исходная матрица.
        :return: Вектор средних значений.
        """
        mean_vector = Matrix(f"{X.num_columns} 1")
        for i in range(1, X.num_columns + 1):
            mean_vector[i, 1] = sum(X[j, i] for j in range(1, X.num_rows + 1)) / X.num_rows

        return mean_vector

    def center_data(X: 'Matrix') -> 'Matrix':
        """
        Центрирование данных в матрице X.

        :param X: Матрица, которую нужно центрировать.
        :return: Центрированная матрица.
        """
        mean_X_vector = PCA.mean_vector(X)
        X_centered = Matrix(f"{X.num_rows} {X.num_columns}")
        for i in range(1, X.num_rows + 1):
            for j in range(1, X.num_columns + 1):
                X_centered[i, j] = X[i, j] - mean_X_vector[j, 1]

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

    def find_eigenvalues(covariance_matrix: 'Matrix',
                         tolerance: Optional[float] = 1e-10,
                         initial_num_intervals: Optional[int] = 10,
                         max_num_intervals: Optional[int] = 100) -> List[float]:
        """
        Вычисление собственных значений ковариационной матрицы.

        :param covariance_matrix: Ковариационная матрица.
        :param tolerance: Допустимая погрешность для вычисления собственных значений.
        :param initial_num_intervals: Изначальное число интервалов для поиска на разбиении.
        :param max_num_intervals: Максимальное число интервалов для поиска на разбиении.
        :return: Собственные значения.

        Осторожно, при низкой точности нахождения ковариационной матрицы, ответ не находится из-за изначальных погрешностей.
        """
        eigenvalues = []
        matrix = deepcopy(covariance_matrix)

        zero_eigenvalue = \
            (PCA.characteristic_polynomial_value(matrix, -tolerance)
             * PCA.characteristic_polynomial_value(matrix, tolerance) <= 0
             or PCA.characteristic_polynomial_value(matrix, 0) < 1e-6)
        one_eigenvalue = abs(PCA.characteristic_polynomial_value(matrix, 1)) < tolerance
        eigenvalues.append(1)

        num_intervals = initial_num_intervals

        while len(eigenvalues) - 1 != matrix.num_rows - zero_eigenvalue and num_intervals <= max_num_intervals:
            current_eigenvalues = deepcopy(eigenvalues)
            for interval_index in range(len(current_eigenvalues) + 1):
                if interval_index == 0:
                    lower_bound = tolerance
                    higher_bound = current_eigenvalues[0] - tolerance
                elif interval_index == len(current_eigenvalues):
                    lower_bound = current_eigenvalues[-1] + tolerance
                    higher_bound = matrix.get_tracer() + tolerance
                else:
                    lower_bound = current_eigenvalues[interval_index - 1] + tolerance
                    higher_bound = current_eigenvalues[interval_index] - tolerance
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
                    characteristic_value_low = PCA.characteristic_polynomial_value(matrix, low)
                    characteristic_value_high = PCA.characteristic_polynomial_value(matrix, high)

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
                            characteristic_value = PCA.characteristic_polynomial_value(matrix, eigenvalue)
                            if abs(high - low) < tolerance:
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

    def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
        """
        Вычисление доли абсолютной дисперсии для первых k собственных значений.

        :param eigenvalues: Собственные значения.
        :param k: Количество собственных значений.
        :return: Доля абсолютной дисперсии.
        """

        return sum(eigenvalues[-k:]) / sum(eigenvalues)

    def gauss_solver(A: 'Matrix', b: 'Matrix') -> List['Matrix']:
        order = list(range(A.num_rows))

        for i in range(A.num_rows - 1):
            max_value = 0.0
            col = i - 1
            while round(abs(max_value), A.accuracy) == 0.0 and col != A.num_rows:
                col += 1
                for j in range(i, A.num_rows):
                    value = A[order[j] + 1, col + 1]
                    if abs(value) > abs(max_value):
                        max_value = value
                        pivot_row = j

            if round(max_value, A.accuracy) == 0.0:
                continue

            if order[pivot_row] != i:
                order[i], order[pivot_row] = order[pivot_row], order[i]

            for j in range(i + 1, A.num_rows):
                factor = A[order[j] + 1, col + 1] / max_value
                if round(factor, A.accuracy) == 0.0:
                    continue

                for k in range(col, A.num_rows):
                    A[order[j] + 1, k + 1] -= factor * A[order[i] + 1, k + 1]

                b[order[j] + 1, 1] -= factor * b[order[i] + 1, 1]

        A.accuracy = 5
        amt_null_rows = 0
        mn_null_row = A.num_rows
        for i in range(A.num_rows):
            flag_null_row = True
            for j in range(i, A.num_columns):
                if round(A[order[i] + 1, j + 1], A.accuracy) != 0.0:
                    flag_null_row = False
                    break
            if flag_null_row:
                mn_null_row = min(i, mn_null_row)
                amt_null_rows += 1

        if amt_null_rows == 0:
            Ans = Matrix(f"{A.num_rows} 1")
            Ans[A.num_rows, 1] = b[order[A.num_rows - 1] + 1, 1] / A[order[A.num_rows - 1] + 1, A.num_columns]

            for i in range(A.num_rows - 2, -1, -1):
                value = 0

                for j in range(A.num_columns, i + 1, -1):
                    value += A[order[i] + 1, j] * Ans[j, 1]

                Ans[i + 1, 1] = (b[order[i] + 1, 1] - value) / A[order[i] + 1, i + 1]

            basis = []
            basis.append(Ans)

        else:
            null_on_diag = []
            last_ind = A.num_rows
            for i in range(mn_null_row - 1, -1, -1):
                for j in range(i, last_ind):
                    if round(A[order[i] + 1, j + 1], A.accuracy) != 0.0:
                        for k in range(j + 2, last_ind + 1):
                            null_on_diag.append(k - 1)
                        last_ind = j
                        break

            basis = []
            for i in range(amt_null_rows):
                basis.append(Matrix(f"{A.num_rows} 1"))

            j = 0
            for i in null_on_diag:
                basis[j][i + 1, 1] = 1
                j += 1

            for k in range(amt_null_rows):
                for i in range(null_on_diag[k] - 1, -1, -1):
                    if i in null_on_diag:
                        continue

                    first_not_null = A.num_columns
                    for j in range(i, A.num_columns):
                        if round(A[order[i] + 1, j + 1]) != 0.0:
                            first_not_null = j
                            break

                    value = 0
                    for j in range(A.num_columns - 1, first_not_null, -1):
                        value += A[order[i] + 1, j + 1] * basis[k][j + 1, 1]

                    if value != 0:
                        basis[k][i + 1, 1] = (-value) / A[order[i] + 1, first_not_null + 1]

        return basis

    def find_eigenvectors(C: 'Matrix', eigenvalues: List[float]) -> Dict[float, List['Matrix']]:
        """
        Находит собственные векторы для заданной ковариационной матрицы и собственных значений.

        :param covariance_matrix: Ковариационная матрица.
        :param eigenvalues: Собственные значения.
        :return: Словарь собственных значений и соответствующих им собственных векторов.
        """
        eigenvectors = {}
        b = Matrix(f"{C.num_rows} 1")
        I = Matrix.eye(C.num_rows)

        for eigenvalue in eigenvalues:
            A = C - (float(eigenvalue) * I)
            eigenvectors[eigenvalue] = PCA.gauss_solver(A, b)

        return eigenvectors

    def get_components(eigenvectors: Dict[float, List['Matrix']], k: int) -> 'Matrix':
        """
        Получает матрицу компонент из собственных векторов.

        :param eigenvectors: Словарь собственных значений и соответствующих им собственных векторов.
        :param k: Количество компонент.
        :return: Матрица компонент.
        """
        components = Matrix(f"{eigenvectors[next(iter(eigenvectors))][0].num_rows} {k}")
        count = 0
        for eigenvalue in sorted(eigenvectors.keys(), reverse=True):
            if count == k:
                break
            for vector in eigenvectors[eigenvalue]:
                for element in range(1, vector.num_rows + 1):
                    components[element, count + 1] = vector[element, 1]
                count += 1
                if count == k:
                    break

        return components

    def RSA(X: 'Matrix', k: int) -> Tuple['Matrix', 'Matrix', float]:
        """
        Полный алгоритм PCA.

        :param X: Исходная матрица.
        :param k: Число компонент.
        :return: Проекция матрицы X на k компонент, матрица компонент, доля абсолютной дисперсии.
        """
        centered_X = PCA.center_data(X)
        covariance_matrix = PCA.covariance_matrix(centered_X)
        eigenvalues = PCA.find_eigenvalues(covariance_matrix)
        eigenvectors = PCA.find_eigenvectors(covariance_matrix, eigenvalues)
        for eigenvalue in eigenvectors.keys():
            eigenvectors[eigenvalue] = [vector.norm_vector() for vector in eigenvectors[eigenvalue]]
        components = PCA.get_components(eigenvectors, k)
        projection = centered_X * components
        variance = PCA.explained_variance_ratio(eigenvalues, k)

        return projection, components, variance

    def plot_pca_projection(X_proj: 'Matrix') -> Figure:
        """
        Визуализирует проекцию данных на первые две главные компоненты.

        :param X_proj: Проекция данных на главные компоненты.
        :return: Matplotlib Figure.
        """
        x = [X_proj[i, 1] for i in range(1, X_proj.num_rows + 1)]
        y = [X_proj[i, 2] for i in range(1, X_proj.num_rows + 1)]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, s=30, color='steelblue', alpha=0.7, edgecolors='k', linewidths=0.5)
        for i in range(len(x)):
            ax.text(x[i] + 5, y[i], str(i + 1), fontsize=9, color='darkred')
        ax.set_title("Проекция на первые 2 главные компоненты", fontsize=14)
        ax.set_xlabel("Главная компонента 1", fontsize=12)
        ax.set_ylabel("Главная компонента 2", fontsize=12)
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

        return fig

    def reconstruction(projection: 'Matrix', components: 'Matrix', mean_vector: 'Matrix') -> 'Matrix':
        """
        Восстанавливает исходную матрицу из проекции и матрицы компонент.

        :param projection: Проекция матрицы.
        :param components: Матрица компонент.
        :param mean_vector: Вектор средних значений столбцов.
        :return: Восстановленная матрица.
        """
        X_recon = projection * components.transpose()
        for i in range(1, X_recon.num_rows + 1):
            for j in range(1, X_recon.num_columns + 1):
                X_recon[i, j] += mean_vector[j, 1]

        return X_recon

    def reconstruction_error(X_orig: 'Matrix', X_recon: 'Matrix') -> float:
        """
        Вычисляет ошибку восстановления между оригинальной и восстановленной матрицами.

        :param X_orig: Оригинальная матрица.
        :param X_recon: Восстановленная матрица.
        :return: Ошибка восстановления (MAE).
        """
        error = 0
        for i in range(1, X_orig.num_rows + 1):
            for j in range(1, X_orig.num_columns + 1):
                error += (X_orig[i, j] - X_recon[i, j]) ** 2

        return error / (X_orig.num_rows * X_orig.num_columns)

    def auto_select_k(eigenvalues: List[float], threshold: float = 0.95) -> int:
        """
        Автоматически выбирает количество компонент k на основе заданного порога.

        :param eigenvalues: Собственные значения.
        :param threshold: Порог для объясненной дисперсии.
        :return: Выбранное количество компонент k.
        """
        l = 0
        r = len(eigenvalues)
        while l < r:
            m = (l + r) // 2
            if PCA.explained_variance_ratio(eigenvalues, m + 1) >= threshold:
                r = m
            else:
                l = m + 1

        return l + 1

    def handle_missing_values(dataset: str) -> 'Matrix':
        """
        Обрабатывает пропущенные значения в наборе данных.
        Заменяет пропущенные значения на средние значения соответствующих колонок.

        :param dataset: Набор данных в виде строки.
        :return: Обработанный набор данных в виде матрицы.
        """
        X = Matrix()
        input_string = dataset.split('\n')

        size_string = input_string[0].split()
        X.num_rows = int(size_string[0])
        X.num_columns = int(size_string[1])
        X.square = X.num_rows == X.num_columns

        X.values = []
        X.row_sizes = [0]
        X.column_indices = []
        X.accuracy = int(input_string[1])

        missing_values = []
        columns_sums = [0] * X.num_columns
        columns_sizes = [0] * X.num_columns

        for row in range(X.num_rows):
            row_string = input_string[row + 2].split()
            X.row_sizes.append(X.row_sizes[-1])
            for column in range(X.num_columns):
                element = row_string[column]
                if element == 'nan':
                    missing_values.append((row, column))
                else:
                    element = round(float(element), X.accuracy)
                    if element != 0:
                        X.values.append(element)
                        X.column_indices.append(column)
                        X.row_sizes[-1] += 1
                        columns_sums[column] += element
                        columns_sizes[column] += 1

        columns_means = []
        for sum, size in zip(columns_sums, columns_sizes):
            if size != 0:
                columns_means.append(sum / size)
            else:
                columns_means.append(0)
        for row, column in missing_values:
            X[row + 1, column + 1] = columns_means[column]

        return X

    def add_noise_and_compare(X: 'Matrix', noise_level: float = 0.1):
        """
        Добавляет шум к данным и сравнивает восстановленные данные с оригинальными.

        :param X: Исходная матрица.
        :param noise_level: Уровень шума (доля от стандартного отклонения).
        """
        mean_vector = PCA.mean_vector(X)
        standard_deviation = [0] * X.num_columns
        for i in range(1, X.num_rows + 1):
            for j in range(1, X.num_columns + 1):
                standard_deviation[j - 1] += (X[i, j] - mean_vector[j, 1]) ** 2
        standard_deviation = [math.sqrt(sd / X.num_rows) for sd in standard_deviation]

        noise = Matrix(f"{X.num_rows} {X.num_columns}")
        for i in range(1, X.num_rows + 1):
            for j in range(1, X.num_columns + 1):
                noise[i, j] = random.uniform(-noise_level * standard_deviation[j - 1],
                                             noise_level * standard_deviation[j - 1])

        noisy_X = X + noise
        noisy_centred_X = PCA.center_data(noisy_X)
        noisy_covariance_matrix = PCA.covariance_matrix(noisy_centred_X)
        noisy_eigenvalues = PCA.find_eigenvalues(noisy_covariance_matrix)
        noisy_eigenvectors = PCA.find_eigenvectors(noisy_covariance_matrix, noisy_eigenvalues)
        for noisy_eigenvalue in noisy_eigenvectors.keys():
            noisy_eigenvectors[noisy_eigenvalue] = [vector.norm_vector() for vector in
                                                    noisy_eigenvectors[noisy_eigenvalue]]
        noisy_eigenvectors_count = sum([len(v) for v in noisy_eigenvectors.values()])
        noisy_mean_vector = PCA.mean_vector(noisy_X)

        centered_X = PCA.center_data(X)
        covariance_matrix = PCA.covariance_matrix(centered_X)
        eigenvalues = PCA.find_eigenvalues(covariance_matrix)
        eigenvectors = PCA.find_eigenvectors(covariance_matrix, eigenvalues)
        for eigenvalue in eigenvectors.keys():
            eigenvectors[eigenvalue] = [vector.norm_vector() for vector in eigenvectors[eigenvalue]]
        eigenvectors_count = sum([len(v) for v in eigenvectors.values()])
        mean_vector = PCA.mean_vector(X)

        for k in range(1, min(eigenvectors_count, noisy_eigenvectors_count) + 1):
            noisy_components = PCA.get_components(noisy_eigenvectors, k)
            noisy_projection = noisy_centred_X * noisy_components
            noisy_reconstructed_X = PCA.reconstruction(noisy_projection, noisy_components, noisy_mean_vector)
            noisy_error_from_noisy = PCA.reconstruction_error(noisy_X, noisy_reconstructed_X)
            noisy_error = PCA.reconstruction_error(X, noisy_reconstructed_X)

            components = PCA.get_components(eigenvectors, k)
            projection = centered_X * components
            reconstructed_X = PCA.reconstruction(projection, components, mean_vector)
            error = PCA.reconstruction_error(X, reconstructed_X)

            print(f"Компоненты: {k}, Ошибка восстановления (оригинал): {error}, "
                  f"Ошибка восстановления (шум): {noisy_error}, "
                  f"Ошибка восстановления (шум с шумом): {noisy_error_from_noisy}")

    def apply_pca_to_dataset(dataset_name: str, k: int) -> Tuple['Matrix', float]:
        """
        Применяет PCA к заданному набору данных и возвращает проекцию и ошибку.

        :param dataset_name: Имя набора данных.
        :param k: Количество компонент.
        :return: Проекция и ошибка восстановления.
        """
        file = open(f"tests/{dataset_name}.txt", "r")
        ds = file.read()
        file.close()
        X = PCA.handle_missing_values(ds)

        projection, components, variance = PCA.RSA(X, k)
        mean_vector = PCA.mean_vector(X)

        reconstructed_X = PCA.reconstruction(projection, components, mean_vector)
        error = PCA.reconstruction_error(X, reconstructed_X)

        return projection, error


if __name__ == '__main__':
    # m = Matrix(
    #     "3 3\n"
    #     "100\n"
    #     "1 2 3\n"
    #     "4 5 6\n"
    #     "7 8 9\n")
    # m = PCA.center_data(m)
    # m = PCA.covariance_matrix(m)
    # print(m)
    # eigenvalues = PCA.find_eigenvalues(m)

    # eigenvalues = [0, 0.0001259685, 0.0236764174, 302.3489186721, 389.8680825373, 1513.9050431088, 63252.0450021848]
    # eigenvectors = PCA.find_eigenvectors(m, eigenvalues)
    #
    m = Matrix(
        "9 8\n"
        "100\n"
        "2 4 54.5 3.5 2 4 54.6 3.5\n"
        "23 43 45 56 2 4 54.5 3.5\n"
        "34 45 56 67 2 4 54.9 3.5\n"
        "213 94.5 35 34 2 4 54.21 3.5\n"
        "39 24 59 34 2 4 54.64 3.5\n"
        "23 45 56 67 2 4 54.3 3.5\n"
        "45 95 95 45 2 4 54.5643 3.5\n"
        "777 33 43.2 45.4 2 4 54.556 3.5\n"
        "2.1 2.3 6.5 4.5 2 4 54.5 3.5564\n")
    # projection, components, variance = PCA.RSA(m, 4)
    # mean_vector = PCA.mean_vector(m)
    # reconstructed_m = PCA.reconstruction(projection, components, mean_vector)
    #
    # error = PCA.reconstruction_error(m, reconstructed_m)
    # print(error)

    # projection, error = PCA.apply_pca_to_dataset('pokemon', 3)
    # print(error)

    # PCA.add_noise_and_compare(m, 0.1)
