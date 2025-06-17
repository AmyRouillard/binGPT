# %%

import sympy as sp

# Declare symbols
w11, w12, w21, w22 = sp.symbols("w_1^1 w_1^2 w_2^1 w_2^2")
b11, b21 = sp.symbols("b^1_1 b^2_1")
lmbda = sp.symbols("lambda")


# a_{ij} as a[i][j]
a = [[sp.symbols(f"a_{i}{j}") for j in range(3)] for i in range(3, 0, -1)]

# Helper access
a_ij = lambda i, j: a[3 - i][j]  # so a_ij(1,1) → a[2][1] = a_11

# Define H row-by-row
H = sp.Matrix(
    [
        [
            -w12 * w12 * a_ij(1, 2),
            -w12 * w22 * a_ij(3, 2),
            -w12 * w12 * a_ij(1, 1),
            # -w12 * w22 * a_ij(3, 1),
            # -w12 * (w11 * a_ij(1, 2) + b11 * a_ij(1, 1)),
            # -w12 * (w21 * a_ij(3, 2) + b21 * a_ij(3, 1)),
            # -w12 * a_ij(1, 1),
        ],
        [
            -w22 * w12 * a_ij(3, 2),
            -w22 * w22 * a_ij(2, 2),
            -w22 * w12 * a_ij(3, 1),
            # -w22 * w22 * a_ij(2, 1),
            # -w22 * (w11 * a_ij(3, 2) + b11 * a_ij(3, 1)),
            # -w22 * (w21 * a_ij(2, 2) + b21 * a_ij(2, 1)),
            # -w22 * a_ij(2, 1),
        ],
        [
            -w12 * w12 * a_ij(1, 1),
            -w12 * w22 * a_ij(3, 1),
            -w12 * w12 * a_ij(1, 0),
            # -w12 * w22 * a_ij(3, 0),
            # -w12 * (w11 * a_ij(1, 1) + b11 * a_ij(1, 0)),
            # -w12 * (w21 * a_ij(3, 1) + b21 * a_ij(3, 0)),
            # -w12 * a_ij(1, 0),
        ],
        # [
        #     -w22 * w12 * a_ij(3, 1),
        #     -w22 * w22 * a_ij(2, 1),
        #     -w22 * w12 * a_ij(3, 0),
        #     -w22 * w22 * a_ij(2, 0),
        #     -w22 * (w11 * a_ij(3, 1) + b11 * a_ij(3, 0)),
        #     -w22 * (w21 * a_ij(2, 1) + b21 * a_ij(2, 0)),
        #     -w22 * a_ij(2, 0),
        # ],
        # [
        #     -w11 * w12 * a_ij(1, 2) - b11 * w12 * a_ij(1, 1),
        #     -w11 * w22 * a_ij(3, 2) - b11 * w22 * a_ij(3, 1),
        #     -w11 * w12 * a_ij(1, 1) - b11 * w12 * a_ij(1, 0),
        #     -w11 * w22 * a_ij(3, 1) - b11 * w22 * a_ij(3, 0),
        #     -w11 * (w11 * a_ij(1, 2) + b11 * a_ij(1, 1))
        #     - b11 * (w11 * a_ij(1, 1) + b11 * a_ij(1, 0)),
        #     -w11 * (w21 * a_ij(3, 2) + b21 * a_ij(3, 1))
        #     - b11 * (w21 * a_ij(3, 1) + b21 * a_ij(3, 0)),
        #     -w11 * a_ij(1, 1) - b11 * a_ij(1, 0),
        # ],
        # [
        #     -w21 * w12 * a_ij(3, 2) - b21 * w12 * a_ij(3, 1),
        #     -w21 * w22 * a_ij(2, 2) - b21 * w22 * a_ij(2, 1),
        #     -w21 * w12 * a_ij(3, 1) - b21 * w12 * a_ij(3, 0),
        #     -w21 * w22 * a_ij(2, 1) - b21 * w22 * a_ij(2, 0),
        #     -w21 * (w11 * a_ij(3, 2) + b11 * a_ij(3, 1))
        #     - b21 * (w11 * a_ij(3, 1) + b11 * a_ij(3, 0)),
        #     -w21 * (w21 * a_ij(2, 2) + b21 * a_ij(2, 1))
        #     - b21 * (w21 * a_ij(2, 1) + b21 * a_ij(2, 0)),
        #     -w21 * a_ij(2, 1) - b21 * a_ij(2, 0),
        # ],
        # [
        #     -w12 * a_ij(1, 1),
        #     -w22 * a_ij(2, 1),
        #     -w12 * a_ij(1, 0),
        #     -w22 * a_ij(2, 0),
        #     -(w11 * a_ij(1, 1) + b11 * a_ij(1, 0)),
        #     -(w21 * a_ij(2, 1) + b21 * a_ij(2, 0)),
        #     -1,
        # ],
    ]
)


# Identity matrix
I = sp.eye(H.shape[0])

# Compute eigenvalues of H - 2λ I
H_shifted = -H - lmbda * I

# %%

# set a_ij(i,0) to 1
for i in range(1, 4):
    # substitute a_ij(i,0) with 1
    H_shifted = H_shifted.subs(a_ij(i, 0), 1)
for i in range(1, 4):
    # substitute a_ij(i,0) with 1
    H_shifted = H_shifted.subs(a_ij(i, 1), 1 / 2)
for i in range(1, 4):
    # substitute a_ij(i,0) with 1
    H_shifted = H_shifted.subs(a_ij(i, 2), 1 / 3)

# %%

# characteristic polynomial
char_poly = H_shifted.det()
# Print the characteristic polynomial
print("Characteristic Polynomial:")
print(sp.simplify(char_poly))

# %%

# solve the characteristic polynomial for eigenvalues
eigenvals = sp.solve(char_poly, lmbda)

# %%
eigenvals = H_shifted.eigenvals()

# Print the symbolic eigenvalues
for val, mult in eigenvals.items():
    print(f"Eigenvalue: {val}, Multiplicity: {mult}")

# %%
