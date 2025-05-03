import numpy as np

"""
a) (*) Erzeugen Sie ein 1D-Numpy-Array mit Ganzzahlen von 5 bis 100 (einschließlich 100), die
durch 5 teilbar sind (geht in einer Zeile).
"""
print(f"a)\n {np.array([num for num in range(0, 101) if num % 5 == 0])}\n\n")

"""
b) (*) Kehren Sie die Reihenfolge der Elemente in einem gegebenen Vektor um (verwenden Sie Array-
Slicing. Lesen Sie sich bitte ein, was das ist - https://numpy.org/doc/2.2/user/basics.
indexing.html).
"""
print(f"b)\n {np.array([num for num in range(0, 101) if num % 5 == 0])[::-1]}\n\n")

"""
c) (*) Erzeugen Sie eine 5x5-Matrix mit Ganzzahlen von 0 bis 24, wobei die Zahlen zeilenweise
gefüllt werden (links oben bis rechts unten).
"""
print(f"c)\n {np.arange(25).reshape((5, 5))}\n\n")

"""
d) (*) Erzeugen Sie eine 6x6-Matrix mit Zufallswerten im Bereich [0, 100). Finden Sie das Minimum,
das Maximum und normalisieren Sie die Matrix, sodass alle Werte im Bereich [0, 1] liegen.
"""
mat_d = np.random.randint(0, 100, size=(6, 6))
mat_d_min = np.min(mat_d)
mat_d_max = np.max(mat_d)
mat_d_normalized = (mat_d - mat_d_min) / (mat_d_max - mat_d_min)

print("d)")
print("Random Matix:\n", mat_d)
print("Min value: ", mat_d_min)
print("Max value: ", mat_d_max)
print("Normized Matrix: \n", mat_d_normalized, "\n\n")

"""
e) (*) Multiplizieren Sie eine 5x3-Matrix mit einer 3x4-Matrix. Was ist die Form (Shape: row:column)
der resultierenden Matrix?
"""
mat_e_1 = np.random.randint(0, 100, size=(5, 3))
mat_e_2 = np.random.randint(0, 100, size=(3, 4))
mat_e_mult = mat_e_1 @ mat_e_2
print("e)\n Shape of multiplied matrix: ", mat_e_mult.shape, "\n\n")

"""
f) (*) Erstellen Sie ein Numpy-Array mit Werten von -5 bis 15. Erzeugen Sie ein neues Array, das
nur die positiven geraden Zahlen enthält. Verwenden Sie boolesche Indizierung.
"""
mat_f = np.arange(-5, 16)
print(f"f) \nMatrix with values from -5 to 15: \n{mat_f}\n"
      f"Matrix w boolean filter for number > 0 and even: \n{mat_f[(mat_f > 0) & (mat_f % 2 == 0)]}\n\n")

"""
g) (*) Erzeugen Sie eine 6x6-Matrix mit den Werten 1 bis 36 und extrahieren Sie daraus eine 2x2-
Untermatrix aus der Mitte der Matrix (Array-Slicing again).
"""
mat_g = np.arange(1, 37).reshape(6, 6)
print(f"g)\nNew Matrix w values from 1 to 36: \n{mat_g}")
print(f"Sub-matrix: \n{mat_g[1:5, 1:5]}\n\n")

"""
h) (*) Erzeugen Sie ein 1D-Numpy-Array mit Werten von 0 bis 30. Ersetzen Sie alle Werte zwischen
10 und 20 (einschließlich) durch ihr Negatives.
"""
mat_h = np.arange(31)
mat_h_2 = mat_h.copy()
mat_h_2[(mat_h_2 >= 10) & (mat_h_2 <= 20)] *= -1
print(f"h)\nArray with numbers from 0 to 30:\n{mat_h}")
print(f"Array with inverted values from 10 to 20:\n{mat_h_2}\n\n")

"""
i) (*) Erzeugen Sie ein zweidimensionales Array mit Zufallswerten und berechnen Sie die Summe
aller Elemente sowie die Summe entlang der Achsen (Zeilen- und Spaltensummen).
"""

mat_i = np.random.randint(0, 10, size=(3, 3))
print(f"i)\nMatrix to sum rows and columns from:\n{mat_i}\n"
      f"Sum of rows:\n{mat_i.sum(axis=1)}\n"
      f"Sum of columns:\n{mat_i.sum(axis=0)}\n\n")

"""
j) (*) Erzeugen Sie eine Matrix M der Größe 3x5 und einen Zeilenvektor v der Größe 5. Multiplizieren
Sie v mit M (Achten Sie auf die richtige Form!).
"""
mat_j = np.arange(1, 16).reshape(3, 5)
vec_j = np.array([1, 2, 3, 4, 5])
print(f"j)\nMatrix:\n{mat_j}\nVector:\n{vec_j}")
print("Either the vector has to be turned into a column vector and the calculation has to be m @ v, or the matrix has to be transposed")
print(f"M @ v with column vector:\n{mat_j @ vec_j.reshape(5, 1)}")
print(f"v @ M with transposed Matrix:\n{vec_j @ mat_j.T}\n\n")


"""
k) (**) Erzeugen Sie eine 6x6-Matrix mit fortlaufenden Werten von 1 bis 36. Geben Sie die Werte
aller ungeraden Spalten (Index 1, 3, 5) aus.
"""
mat_k = np.arange(1, 37).reshape(6, 6)
print(f"k)\nMatrix:\n{mat_k}\n"
      f"Odd coluns of matrix:\n{mat_k[:, 1::2]}\n\n")

"""
l) (**) Erzeugen Sie eine 2D-Punktwolke mit 100 Punkten und setzen Sie dafür einen festen Zufalls-
Seed (np.random.seed(42)). Berechnen Sie den Mittelpunkt und anschließend die Kovarianzmatrix
der zentrierten Daten.
"""
np.random.seed(42)

mat_l = np.random.randint(0, 100, size=(100, 2))
centered_points = mat_l - np.mean(mat_l, axis=0)
cov_matrix = np.cov(centered_points, rowvar=False)

print(f"l)\nCenter point:\n{np.mean(mat_l, axis=0)}")
print(f"Covariance matrix:\n{cov_matrix}\n\n")


"""
m) (**) Erzeugen Sie 50 zufällige 2D-Koordinaten im kartesischen System (10x2 Matrix). Wandeln
Sie diese Koordinaten in Polarkoordinaten um (Radius r und Winkel θ).
"""
mat_m = np.random.randint(-50, 50, size=(10, 2))

complex_points = mat_m[:, 0] + 1j * mat_m[:, 1]
r = np.abs(complex_points)
theta = np.angle(complex_points, deg=True)

mat_m_polar = np.column_stack((r, theta))
print(f"m)\nRandom matrix (10x2):\n{mat_m}")
print(f"Polar coordinates:\n{mat_m_polar}\n\n")

"""
n) (**) Erzeugen Sie zwei passende Arrays für eine Matrixmultiplikation mit dem @-Operator und
führen Sie die Operation durch. Überprüfen Sie das Ergebnis mit np.dot().
"""
mat_n_1 = np.arange(1,7).reshape(2, 3)
mat_n_2 = np.arange(1,7).reshape(3, 2)
mat_n_mult = mat_n_1 @ mat_n_2
mat_n_mult_2 = np.dot(mat_n_1, mat_n_2)
print(f"n)\nMatrices:\n{mat_n_1}\n{mat_n_2}")
print(f"Result using @:\n{mat_n_mult}\n"
      f"Result using np.dot()\n{mat_n_mult_2}\n\n")
