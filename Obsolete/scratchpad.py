import numpy as np
import numba

@numba.jit(nopython=True)
def test(a, b):
    c = np.zeros(2)
    dot_product = np.dot(c, b)
    angle = np.arccos(dot_product)
    return angle

vector_2 = [1, 0]

#unit_vector_1 = vector_1 / np.linalg.norm(vector_1)

unit_vector_2 = vector_2 / np.linalg.norm(vector_2)

unit_vector_1 = np.zeros(2)
print(type(unit_vector_1))
print(unit_vector_1)
print(type(unit_vector_2))
print(unit_vector_2)
print(test(unit_vector_1, unit_vector_2))
