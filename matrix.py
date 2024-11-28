class Matrix:
    num_rows: int
    num_columns: int
    square: bool
    tracer: float

    values: list[int]
    row_sizes: list[int]
    column_indices: list[int]

    def __init__(self, input_string: str):
        input_string = [row.split() for row in input_string.split('\n')]
        matrix_list: list[list[float]] = [[float(value) for value in row] for row in input_string[1:]]

        self.num_rows = int(input_string[0][0])
        self.num_columns = int(input_string[0][1])
        self.square = self.num_rows == self.num_columns

        self.values = []
        self.row_sizes = [0]
        self.column_indices = []

        for row in range(self.num_rows):
            self.row_sizes.append(self.row_sizes[-1])

            for column in range(self.num_columns):
                if matrix_list[row][column] != 0:
                    self.values.append(matrix_list[row][column])
                    self.column_indices.append(column)
                    self.row_sizes[-1] += 1

    def __getitem__(self, item: [int, int]) -> float:
        row, column = item[0] - 1, item[1] - 1
        if column in self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]]:
            return self.values[
                self.row_sizes[row] +
                self.column_indices[self.row_sizes[row]:self.row_sizes[row + 1]].index(column)
                ]
        else:
            return 0

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.num_rows != other.num_rows or self.num_columns != other.num_columns:
            raise Exception('Matrices must have the same size')

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
