from matrix import Matrix


class PCA():
    def center_data(X: 'Matrix') -> 'Matrix':
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
        X_centered_transposed = X_centered.transpose()
        X_centered_transposed.accuracy = X_centered.accuracy * 2
        covariance_matrix = X_centered_transposed * X_centered
        covariance_matrix.accuracy = X_centered.accuracy
        covariance_matrix /= X_centered.num_rows - 1
        return covariance_matrix


if __name__ == '__main__':
    m = Matrix("3 3\n"
               "2\n"
               "1 0 0\n"
               "0 0 0\n"
               "-7 8.54 0.45")
    centred_m = PCA.covariance_matrix(PCA.center_data(m))
    print(centred_m.get_list())
