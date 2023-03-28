# import numpy as np
# from cvxopt import matrix, solvers

# c = matrix([-6., -4., -5.])

# G = matrix([[ 16., 7.,  24.,  -8.,   8.,  -1.,  0., -1.,  0.,  0.],
#                 [-14., 2.,   7., -13., -18.,   3.,  0.,  0., -1.,  0.],
#                 [  5., 0., -15.,  12.,  -6.,  17.,  0.,  0.,  0., -1.]])

# G = np.array(G)
# G = matrix(G)

# h = matrix( [ -3., 5.,  12.,  -2., -14., -13., 10.,  0.,  0.,  0.] )

# dims = {'l': 2, 'q': [4 for i in range(2)], 's': []}

# # P = np.zeros()
# sol = solvers.conelp(c, G, h, dims)

class MyClass:
    def __init__(self):
        self.attr = "my attribute"
        self.add_new_attribute("newwwww !")
    def add_new_attribute(self, s_input):
        self.s = s_input


myclass = MyClass()
# myclass.add_new_attribute("newwwww !")
print(myclass.s)