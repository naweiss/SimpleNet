class Matrix():
    def __init__(self, n, m, data):
        self.n = n
        self.m = m
        self.data = data
            
    def transpose(self):
        """
        transpose Matrix
        """
        new_data = []
        for i in range(self.m):
            new_row = []
            for j in range(self.n):
                new_row.append(self.data[j][i])
            new_data.append(new_row)                
        return Matrix(self.m, self.n, new_data)

    def apply(self, func):
        """ 
        Element wise function mutation
        x = f(x)
        """
        if callable(func):
            new_data = list(map(lambda row: list(map(lambda col: func(col),row)),self.data))
            return Matrix(self.n, self.m, new_data)
        raise Exception("Variable must be callable")

    def __helper(row1, row2):
        """
        Element wise multiplication of lists
        """
        return sum(map(lambda x: x[0]*x[1],zip(row1,row2)))

    def __repr__(self):
        """ 
        Pretty print matrix
        """
        data_str = [[str(cell) for cell in row] for row in self.data]
        lens     = [max(map(len, col)) for col in zip(*data_str)]
        fmt      = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table    = [fmt.format(*row) for row in data_str]
        sizes    = '['+str(self.n)+' x '+str(self.m)+']'
        return '\n'.join([sizes]+table)+'\n'

    def __add__(self, matrix):
        """ 
        Element wise addition
        """
        if isinstance(matrix,Matrix):
            if self.n == matrix.n and self.m == matrix.m:
                new_data = []
                for i in range(self.n):
                    new_row = []
                    for j in range(self.m):
                        x = self.data[i][j]+matrix.data[i][j]
                        new_row.append(x)
                    new_data.append(new_row)
                return Matrix(self.n, self.m, new_data)
            raise Exception("Invalid matrix addition")

    def __sub__(self,matrix):
        """ 
        Element wise substraction
        """
        if isinstance(matrix,Matrix):
            if self.n == matrix.n and self.m == matrix.m:
                new_data = []
                for i in range(self.n):
                    new_row = []
                    for j in range(self.m):
                        x = self.data[i][j]-matrix.data[i][j]
                        new_row.append(x)
                    new_data.append(new_row)
                return Matrix(self.n, self.m, new_data)
            raise Exception("Invalid matrix substraction")

    def __mul__(self, matrix):
        """
        Matrix multiplication
        """
        if isinstance(matrix,Matrix):
            if self.m == matrix.n:
                new_data = []
                other = matrix.transpose()
                for i in range(self.n):
                    new_row = []
                    for j in range(matrix.m):
                        x = Matrix.__helper(self.data[i],other.data[j])
                        new_row.append(x)
                    new_data.append(new_row)
                return Matrix(self.n, matrix.m, new_data)
            raise Exception("Invalid matrix multiplication")
        
    def __pow__(self, matrix):
        """ 
        Hadamard product
        Element wise multiplication
        """
        if isinstance(matrix,Matrix):
            if self.n == matrix.n and self.m == matrix.m:
                new_data = []
                for i in range(self.n):
                    new_row = []
                    for j in range(matrix.m):
                        new_row.append(self.data[i][j]*matrix.data[i][j])
                    new_data.append(new_row)
                return Matrix(self.n, self.m, new_data)
            raise Exception("Invalid matrix hadamard product")
        
if __name__ == "__main__":
    a = [[1,2],[3,4]]
    b = [[5,6,8],[1,8,9]]
    x = Matrix(2,2,a)
    y = Matrix(2,3,b)
    print(x)
    print(y)
    print(x*y)
    print(x**x)
