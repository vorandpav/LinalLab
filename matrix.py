from copy import deepcopy


class Matrix:
    accuracy: int = 100
    num_rows: int
    num_columns: int
    square: bool

    values: list[float]
    row_sizes: list[int]
    column_indices: list[int]

    def __init__(self, input_string: str = None):
        if input_string is None:
            return

        input_string = input_string.split('\n')

        size_string = input_string[0].split()
        self.num_rows = int(size_string[0])
        self.num_columns = int(size_string[1])
        self.square = self.num_rows == self.num_columns

        self.values = []
        self.row_sizes = [0]
        self.column_indices = []

        if len(input_string) == 1:
            for row in range(self.num_rows):
                self.row_sizes.append(0)
        else:
            self.accuracy = int(input_string[1])

            for row in range(self.num_rows):
                row_string = input_string[row + 2].split()
                self.row_sizes.append(self.row_sizes[-1])

                for column in range(self.num_columns):
                    element = round(float(row_string[column]), self.accuracy)
                    if element != 0:
                        self.values.append(element)
                        self.column_indices.append(column)
                        self.row_sizes[-1] += 1

    def __getitem__(self, item: [int, int]) -> float:
        row, column = item[0] - 1, item[1] - 1
        if (len(self.values) != 0
                and column in self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]]):
            return self.values[
                self.row_sizes[row]
                + self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]].index(column)]
        else:
            return 0

    def __setitem__(self, item: [int, int], value: float) -> None:
        row, column = item[0] - 1, item[1] - 1
        value = round(value, self.accuracy)

        if value == 0:
            if (len(self.values) != 0
                    and column in self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]]):
                for index in range(
                        self.row_sizes[row]
                        + self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]]
                                .index(column),
                        self.row_sizes[-1] - 1):
                    self.values[index] = self.values[index + 1]
                    self.column_indices[index] = self.column_indices[index + 1]

                for row_size in range(row + 1, self.num_rows + 1):
                    self.row_sizes[row_size] -= 1

                self.column_indices.pop()
                self.values.pop()

        else:
            if len(self.values) == 0:
                self.values.append(value)
                self.column_indices.append(column)
                for row_size in range(row + 1, self.num_rows + 1):
                    self.row_sizes[row_size] += 1

            elif column in self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]]:
                self.values[
                    self.row_sizes[row]
                    + self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]]
                    .index(column)] = value

            else:
                self.values.append(self.values[-1])
                self.column_indices.append(self.column_indices[-1])
                for index in range(self.row_sizes[-1] - 1, self.row_sizes[row + 1], -1):
                    self.values[index] = self.values[index - 1]
                    self.column_indices[index] = self.column_indices[index - 1]

                for row_size in range(row + 1, self.num_rows + 1):
                    self.row_sizes[row_size] += 1

                self.column_indices[self.row_sizes[row + 1] - 1] = column
                self.values[self.row_sizes[row + 1] - 1] = value

    def _matrices_sum(self, other: 'Matrix') -> 'Matrix':
        result_matrix = Matrix("0 0")
        result_matrix.accuracy = self.accuracy
        result_matrix.num_rows = self.num_rows
        result_matrix.num_columns = self.num_columns
        result_matrix.square = self.square

        for row in range(1, self.num_rows + 1):
            result_matrix.row_sizes.append(result_matrix.row_sizes[-1])

            first_ptr = self.row_sizes[row - 1]
            second_ptr = other.row_sizes[row - 1]

            while first_ptr < self.row_sizes[row] or second_ptr < other.row_sizes[row]:
                if (first_ptr < self.row_sizes[row]
                        and second_ptr < other.row_sizes[row]
                        and self.column_indices[first_ptr] == other.column_indices[second_ptr]):
                    if round(self.values[first_ptr]
                             + other.values[second_ptr], result_matrix.accuracy) != 0:
                        result_matrix.values.append(
                            round(self.values[first_ptr]
                                  + other.values[second_ptr], result_matrix.accuracy))
                        result_matrix.column_indices.append(self.column_indices[first_ptr])
                        result_matrix.row_sizes[-1] += 1

                    first_ptr += 1
                    second_ptr += 1

                elif (first_ptr < self.row_sizes[row]
                      and (second_ptr >= other.row_sizes[row]
                           or self.column_indices[first_ptr] < other.column_indices[second_ptr])):
                    result_matrix.values.append(self.values[first_ptr])
                    result_matrix.column_indices.append(self.column_indices[first_ptr])
                    result_matrix.row_sizes[-1] += 1
                    first_ptr += 1

                else:
                    if round(other.values[second_ptr], result_matrix.accuracy) != 0:
                        result_matrix.values.append(round(other.values[second_ptr], result_matrix.accuracy))
                        result_matrix.column_indices.append(other.column_indices[second_ptr])
                        result_matrix.row_sizes[-1] += 1
                    second_ptr += 1

        return result_matrix

    def _matrix_and_number_sum(self, other: float) -> 'Matrix':
        result_matrix = Matrix("0 0")
        result_matrix.accuracy = self.accuracy
        result_matrix.num_rows = self.num_rows
        result_matrix.num_columns = self.num_columns
        result_matrix.square = self.square

        for row in range(1, self.num_rows + 1):
            result_matrix.row_sizes.append(result_matrix.row_sizes[-1])
            for element in range(self.row_sizes[row - 1], self.row_sizes[row]):
                if round(self.values[element] + other, result_matrix.accuracy) != 0:
                    result_matrix.values.append(
                        round(self.values[element] + other, result_matrix.accuracy))
                    result_matrix.column_indices.append(self.column_indices[element])
                    result_matrix.row_sizes[-1] += 1

        return result_matrix

    def __add__(self, other) -> 'Matrix':
        if isinstance(other, Matrix):
            if self.num_rows != other.num_rows or self.num_columns != other.num_columns:
                raise Exception('Matrices must have the same size')
            return self._matrices_sum(other)

        else:
            return self._matrix_and_number_sum(other)

    def __radd__(self, other) -> 'Matrix':
        return self + other

    def __iadd__(self, other) -> 'Matrix':
        return self + other

    def __sub__(self, other) -> 'Matrix':
        return self + (-1) * other

    def __rsub__(self, other) -> 'Matrix':
        return (-1) * self + other

    def __isub__(self, other) -> 'Matrix':
        return self - other

    def transpose(self) -> 'Matrix':
        column_row = [[] for _ in range(self.num_columns)]
        column_value = [[] for _ in range(self.num_columns)]

        for row in range(1, self.num_rows + 1):
            for element in range(self.row_sizes[row - 1], self.row_sizes[row]):
                column_row[self.column_indices[element]].append(row - 1)
                column_value[self.column_indices[element]].append(self.values[element])

        result_matrix = Matrix("0 0")
        result_matrix.accuracy = self.accuracy
        result_matrix.num_rows = self.num_columns
        result_matrix.num_columns = self.num_rows
        result_matrix.square = self.square

        for column in range(self.num_columns):
            result_matrix.row_sizes.append(result_matrix.row_sizes[-1])
            for element in range(len(column_row[column])):
                result_matrix.values.append(column_value[column][element])
                result_matrix.column_indices.append(column_row[column][element])
                result_matrix.row_sizes[-1] += 1

        return result_matrix

    def _matrices_mul(self, other: 'Matrix') -> 'Matrix':
        other = other.transpose()

        result_matrix = Matrix("0 0")
        result_matrix.accuracy = self.accuracy
        result_matrix.num_rows = self.num_rows
        result_matrix.num_columns = other.num_rows
        result_matrix.square = result_matrix.num_rows == result_matrix.num_columns

        for row in range(1, self.num_rows + 1):
            result_matrix.row_sizes.append(result_matrix.row_sizes[-1])
            for other_row in range(1, other.num_rows + 1):
                element = 0
                first_ptr = self.row_sizes[row - 1]
                second_ptr = other.row_sizes[other_row - 1]

                while first_ptr < self.row_sizes[row] and second_ptr < other.row_sizes[other_row]:
                    if self.column_indices[first_ptr] == other.column_indices[second_ptr]:
                        element += self.values[first_ptr] * other.values[second_ptr]
                        first_ptr += 1
                        second_ptr += 1

                    elif self.column_indices[first_ptr] < other.column_indices[second_ptr]:
                        first_ptr += 1
                    else:
                        second_ptr += 1

                if round(element, result_matrix.accuracy) != 0:
                    result_matrix.values.append(round(element, result_matrix.accuracy))
                    result_matrix.column_indices.append(other_row - 1)
                    result_matrix.row_sizes[-1] += 1

        return result_matrix

    def _matrix_and_number_mul(self, other: float) -> 'Matrix':
        result_matrix = Matrix("0 0")
        result_matrix.accuracy = self.accuracy
        result_matrix.num_rows = self.num_rows
        result_matrix.num_columns = self.num_columns
        result_matrix.square = self.square

        for row in range(1, self.num_rows + 1):
            result_matrix.row_sizes.append(result_matrix.row_sizes[-1])
            for element in range(self.row_sizes[row - 1], self.row_sizes[row]):
                if round(self.values[element] * other, result_matrix.accuracy) != 0:
                    result_matrix.values.append(
                        round(self.values[element] * other, result_matrix.accuracy))
                    result_matrix.column_indices.append(self.column_indices[element])
                    result_matrix.row_sizes[-1] += 1

        return result_matrix

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.num_columns != other.num_rows:
                raise Exception('Matrices must have the same size')
            return self._matrices_mul(other)

        else:
            return self._matrix_and_number_mul(other)

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            raise Exception('Matrix division is not supported')
        else:
            if other == 0:
                raise Exception('Division by zero')
            else:
                return self * (1 / other)

    def __itruediv__(self, other):
        return self / other

    def get_list(self) -> list[list[float]]:
        return [[self[row + 1, column + 1]
                 for column in range(self.num_columns)]
                for row in range(self.num_rows)]

    def get_tracer(self) -> float:
        if not self.square:
            raise Exception('Matrix is not square')

        tracer = 0
        for row in range(self.num_rows):
            tracer += self[row + 1, row + 1]
        return tracer

    def get_determinant(self) -> float:
        if not self.square:
            raise Exception('Matrix is not square')

        self_copy = deepcopy(self)
        n = self.num_rows

        if n == 1:
            return self_copy[1, 1]
        if n == 2:
            return self_copy[1, 1] * self_copy[2, 2] - self_copy[2, 1] * self_copy[1, 2]

        determinant = 1.0

        for i in range(1, n + 1):
            max_row = i
            for k in range(i + 1, n + 1):
                if abs(self_copy[k, i]) > abs(self_copy[max_row, i]):
                    max_row = k

            if self_copy[max_row, i] == 0:
                return 0

            if max_row != i:
                determinant *= -1
                for j in range(1, n + 1):
                    self_copy[i, j], self_copy[max_row, j] = self_copy[max_row, j], self_copy[i, j]

            determinant *= self_copy[i, i]
            pivot = self_copy[i, i]

            for j in range(i, n + 1):
                self_copy[i, j] /= pivot

            for k in range(i + 1, n + 1):
                factor = self_copy[k, i]
                for j in range(i, n + 1):
                    self_copy[k, j] -= factor * self_copy[i, j]

        return determinant

    def print_determinant_and_invertibility(self) -> None:
        determinant = self.get_determinant()

        if determinant == 0:
            print('нет')
        else:
            print('да')

    def __str__(self) -> str:
        return '\n'.join([' '.join(map(str, row)) for row in self.get_list()])

    def norm_vector(self) -> 'Matrix':
        length = 0
        for i in range(self.num_rows):
            length += self[i + 1, 1] ** 2
        length = length ** 0.5
        if length == 0:
            raise Exception('Zero vector')
        result_vector = deepcopy(self)
        for i in range(self.num_rows):
            result_vector[i + 1, 1] = round(result_vector[i + 1, 1] / length, self.accuracy)
        return result_vector

    def eye(size: int, accuracy: int = 100):
        input_lines = [f"{size} {size}", f"{accuracy}"]
        for i in range(size):
            row = ['0'] * size
            row[i] = '1'
            input_lines.append(' '.join(row))
        input_string = '\n'.join(input_lines)

        return Matrix(input_string)


if __name__ == '__main__':
    m = Matrix("3 3\n"
               "2\n"
               "-4 -1 2\n"
               "10 4 -1\n"
               "8 3 1")
    print(m.get_list())
    print("just print\n")

    m = m + 1
    print(m.get_list())
    print("+number\n")

    m = m + Matrix("3 3\n"
                   "2\n"
                   "-1 0 0\n"
                   "0 -5.01 -1\n"
                   "0 0 -2")
    print(m.get_list())
    print("+matrix\n")

    m = m * 2
    print(m.get_list())
    print("*number\n")

    m = m * Matrix("3 3\n"
                   "2\n"
                   "0.5 0 0\n"
                   "0 0.5 0\n"
                   "0 0 0.5")
    print(m.get_list())
    print("*matrix\n")

    print(m.get_tracer())
    print("tracer\n")

    print(m.get_list())
    print("print\n")

    m.print_determinant_and_invertibility()
    print("determinant and invertibility\n")

    print(m.get_list())
    print("print\n")

    m = Matrix("6 6\n"
               "2\n"
               "1 0 0 0 2 0\n"
               "0 0 3 4 0 0\n"
               "0 0 0 0 0 0\n"
               "0 0 0 8 0 5\n"
               "0 0 0 0 0 0\n"
               "0 7 1 0 0 6")
    print(m.get_list())
    print("just print\n")

    print(m.values)
    print("values\n")

    print(m.row_sizes)
    print("row_sizes\n")

    print(m.column_indices)
    print("column_indices\n")

    print(m)
