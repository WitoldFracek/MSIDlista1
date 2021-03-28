import content as cn
import numpy as np
from utils import *
import os
import pickle


x = np.array([2, -1, -2,  0, -3, -6, -10, 5, 1,  8, 10, -14]).reshape(12, 1)
y = np.array([4,  3,  0, 1.5, -1,  2,   0, 5, 3, 10, 15,   0]).reshape(12, 1)

x2 = np.array([-3, -2, -1, 0, 1, 2, 3]).reshape(7, 1)
y2 = np.array([-27, -8, -1, 0, 1, 8, 27]).reshape(7, 1)

d = cn.least_squares(x, y, 3)
print(d)
print()
print()

xs = [i for i in range(3)]
print(xs)




