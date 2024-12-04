class Matrix:
    num_rows: int
    num_columns: int
    square: bool
    tracer: float
    determinant: float

    values: list[float]
    row_sizes: list[int]
    column_indices: list[int]

    def __init__(self, input_string: str):
        input_string = [row.split() for row in input_string.split('\n')]

        self.num_rows = int(input_string[0][0])
        self.num_columns = int(input_string[0][1])
        self.square = self.num_rows == self.num_columns

        self.values = []
        self.row_sizes = [0]
        self.column_indices = []

        if len(input_string) == 1:
            for row in range(self.num_rows):
                self.row_sizes.append(0)
        else:
            matrix_list: list[list[float]] = [[float(value) for value in row] for row in input_string[1:]]

            for row in range(self.num_rows):
                self.row_sizes.append(self.row_sizes[-1])

                for column in range(self.num_columns):
                    if matrix_list[row][column] != 0:
                        self.values.append(matrix_list[row][column])
                        self.column_indices.append(column)
                        self.row_sizes[-1] += 1

    def __getitem__(self, item: [int, int]) -> float:
        row, column = item[0] - 1, item[1] - 1
        if len(self.values) != 0 and column in self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]]:
            return self.values[
                self.row_sizes[row] +
                self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]].index(column)
                ]
        else:
            return 0

    def __setitem__(self, item: [int, int], value: float) -> None:
        row, column = item[0] - 1, item[1] - 1
        if value == 0:
            if len(self.values) != 0 and column in self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]]:
                for index in range(
                        self.row_sizes[row] + self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]].index(
                            column), self.row_sizes[-1] - 1):
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
                    self.row_sizes[row] +
                    self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]].index(column)
                    ] = value

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
        result_matrix = Matrix(f'{self.num_rows} {self.num_columns}')
        for row in range(1, self.num_rows + 1):
            for column in range(1, self.num_columns + 1):
                result_matrix[row, column] = self[row, column] + other[row, column]
        return result_matrix

    def _matrix_and_number_sum(self, other: float) -> 'Matrix':
        result_matrix = Matrix(f'{self.num_rows} {self.num_columns}')
        for row in range(1, self.num_rows + 1):
            for column in range(1, self.num_columns + 1):
                result_matrix[row, column] = self[row, column] + other
        return result_matrix

    def __add__(self, other) -> 'Matrix':
        if isinstance(other, Matrix):
            if self.num_rows != other.num_rows or self.num_columns != other.num_columns:
                raise Exception('Matrices must have the same size')
            return self._matrices_sum(other)

        else:

            return self._matrix_and_number_sum(other)

    def __radd__(self, other):
        return self + other

    def _matrices_mul(self, other: 'Matrix') -> 'Matrix':
        result_matrix = Matrix(f'{self.num_rows} {other.num_columns}')
        for row in range(1, self.num_rows + 1):
            for column in range(1, other.num_columns + 1):
                for i in range(1, self.num_columns + 1):
                    result_matrix[row, column] += self[row, i] * other[i, column]
        return result_matrix

    def _matrix_and_number_mul(self, other: float) -> 'Matrix':
        result_matrix = Matrix(f'{self.num_rows} {self.num_columns}')
        for row in range(1, self.num_rows + 1):
            for column in range(1, self.num_columns + 1):
                result_matrix[row, column] = self[row, column] * other
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

    def get_list(self) -> list[list[float]]:
        return [[self[row + 1, column + 1] for column in range(self.num_columns)] for row in range(self.num_rows)]

    def _calculate_tracer(self) -> None:
        self.tracer = 0
        for row in range(self.num_rows):
            self.tracer += self[row + 1, row + 1]

    def get_tracer(self) -> float:
        if not self.square:
            raise Exception('Matrix is not square')

        self._calculate_tracer()
        return self.tracer

    def _calculate_determinant(self) -> None:
        n = self.num_rows

        if n == 1:
            self.determinant = self[0, 0]
            return
        if n == 2:
            self.determinant = self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
            return

        self.determinant = 1.0

        for i in range(1, n + 1):
            max_row = i
            for k in range(i + 1, n + 1):
                if abs(self[k, i]) > abs(self[max_row, i]):
                    max_row = k

            if self[max_row, i] == 0:
                self.determinant = 0
                return

            if max_row != i:
                self.determinant *= -1
                for j in range(1, n + 1):
                    self[i, j], self[max_row, j] = self[max_row, j], self[i, j]

            self.determinant *= self[i, i]
            pivot = self[i, i]

            for j in range(i, n + 1):
                self[i, j] /= pivot

            for k in range(i + 1, n + 1):
                factor = self[k, i]
                for j in range(i, n + 1):
                    self[k, j] -= factor * self[i, j]

    def get_determinant(self) -> float:
        if not self.square:
            raise Exception('Matrix is not square')

        self._calculate_determinant()
        return self.determinant

    def print_determinant_and_invertibility(self) -> None:
        print(self.get_determinant())

        if self.determinant == 0:
            print('нет')
        else:
            print('да')


if __name__ == '__main__':
    m = Matrix("3 3\n-4 -1 2\n10 4 -1\n8 3 1")
    print(m.get_list())
    print("just print\n")

    m = m + 1
    print(m.get_list())
    print("+number\n")

    m = m + Matrix("3 3\n-1 -1 -1\n-1 -1 -1\n-1 -1 -1")
    print(m.get_list())
    print("+matrix\n")

    m = m * 2
    print(m.get_list())
    print("*number\n")

    m = m * Matrix("3 3\n0.5 0 0\n0 0.5 0\n0 0 0.5")
    print(m.get_list())
    print("*matrix\n")

    print(m.get_tracer())
    print("tracer\n")

    m.print_determinant_and_invertibility()
    print("determinant and invertibility\n")
