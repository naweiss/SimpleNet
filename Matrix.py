import numpy as np

class Matrix:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.n = self.data.shape[0]
        self.m = self.data.shape[1]

    def transpose(self):
        """
        transpose Matrix
        """
        return Matrix(np.transpose(self.data))

    def apply(self, func):
        """ 
        Element wise function mutation: x = f(x)
        """
        if not callable(func):
            raise Exception("Variable must be callable")

        new_data = func(self.data)
        return Matrix(new_data)

    def __repr__(self):
        """ 
        Pretty print matrix
        """
        column_max_lens = [max(map(lambda val: len(str(val)), column)) for column in self.transpose().data]
        padded_data     = [
            [str(cell).ljust(padding) for cell, padding in zip(row, column_max_lens)]
            for row in self.data
        ]
        matrix_str      = "\n".join(
            "\t".join(cell for cell in row)
            for row in padded_data
        )
        return "[{} x {}]\n{}".format(self.n, self.m, matrix_str)

    def __add__(self, matrix):
        """ 
        Element wise addition
        """
        if not isinstance(matrix, Matrix):
            raise Exception("Argument must be a Matrix")
        if self.n != matrix.n or self.m != matrix.m:
            raise Exception("Invalid matrix addition")

        return Matrix(self.data + matrix.data)

    def __sub__(self,matrix):
        """ 
        Element wise substraction
        """
        if not isinstance(matrix, Matrix):
            raise Exception("Argument must be a Matrix")
        if self.n != matrix.n or self.m != matrix.m:
            raise Exception("Invalid matrix substraction")

        return Matrix(self.data - matrix.data)
        
    def __pow__(self, matrix):
        """ 
        Hadamard product
        Element wise multiplication
        """
        if not isinstance(matrix, Matrix):
            raise Exception("Argument must be a Matrix")
        if self.n != matrix.n or self.m != matrix.m:
            raise Exception("Invalid matrix hadamard product")

        return Matrix(self.data * matrix.data)

    @staticmethod
    def __element_wise_mul(row1, row2):
        """
        Element wise multiplication of lists
        """
        return sum(map(lambda x: x[0] * x[1], zip(row1, row2)))

    def __mul__(self, matrix):
        """
        Matrix multiplication
        """
        if not isinstance(matrix, Matrix):
            raise Exception("Argument must be a Matrix")
        if self.m != matrix.n:
            raise Exception("Invalid matrix multiplication")

        return Matrix(np.matmul(self.data, matrix.data))

    def to_list(self):
        return self.data.tolist()


if __name__ == "__main__":
    x = Matrix([
        [1, 2],
        [3, 4]
    ])
    y = Matrix([
        [5, 6, 8],
        [1, 8, 9]
   ])
    print("x: {}".format(x))
    print("y: {}".format(y))
    print("Matrix multiplication of x and y is: {}".format(x * y))
    print("Element wise multiplication of x and x is: {}".format(x ** x))
