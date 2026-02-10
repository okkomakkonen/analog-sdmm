from math import sqrt, pi
from random import shuffle

import numpy as np

from complex_sdmm import circular_normal, compute_noise_variance


def nw(A: np.ndarray) -> np.ndarray:
    """Return the north-west corner of the matrix"""

    n, m = A.shape
    assert n % 2 == 0 and m % 2 == 0
    n2 = n // 2
    m2 = m // 2

    return A[:n2, :m2]


def ne(A: np.ndarray) -> np.ndarray:
    """Return the north-east corner of the matrix"""

    n, m = A.shape
    assert n % 2 == 0 and m % 2 == 0
    n2 = n // 2
    m2 = m // 2

    return A[:n2, m2:]


def sw(A: np.ndarray) -> np.ndarray:
    """Return the south-west corner of the matrix"""

    n, m = A.shape
    assert n % 2 == 0 and m % 2 == 0
    n2 = n // 2
    m2 = m // 2

    return A[n2:, :m2]


def se(A: np.ndarray) -> np.ndarray:
    """Return the south-east corner of the matrix"""

    n, m = A.shape
    assert n % 2 == 0 and m % 2 == 0
    n2 = n // 2
    m2 = m // 2

    return A[n2:, m2:]


def complexify_inner_A(A):
    A_parts = np.split(A, 2, axis=1)

    return A_parts[0] + 1j * A_parts[1]


def complexify_inner_B(B):
    B_parts = np.split(B, 2, axis=0)

    return B_parts[0] - 1j * B_parts[1]


def complexify_inner(A, B):
    """Return complex matrices Ap, Bp such that A @ B == (Ap @ Bp).real

    >>> A = np.random.rand(10, 10)
    >>> B = np.random.rand(10, 10)
    >>> Ap, Bp = complexify_inner(A, B)
    >>> bool(np.isclose(A @ B, (Ap @ Bp).real).all())
    True
    """

    return complexify_inner_A(A), complexify_inner_B(B)


def assemble_outer(P, M):
    return 0.5 * np.block(
        [[P.real + M.real, P.imag - M.imag], [P.imag + M.imag, -P.real + M.real]]
    )


def complexify_outer_A(A):
    A_parts = np.split(A, 2, axis=0)

    return A_parts[0] + 1j * A_parts[1]


def complexify_outer_B(B):
    B_parts = np.split(B, 2, axis=1)

    return B_parts[0] + 1j * B_parts[1]


def complexify_outer(A, B):
    """Return complex matrices Ap, Bp such that

    A @ B == assemble_outer(Ap @ Bp, Ap @ Bp.conjugate())

    >>> A = np.random.rand(10, 10)
    >>> B = np.random.rand(10, 10)
    >>> Ap, Bp = complexify_outer(A, B)
    >>> P, M = Ap @ Bp, Ap @ Bp.conjugate()
    >>> bool(np.isclose(A @ B, assemble_outer(P, M)).all())
    True
    """

    return complexify_outer_A(A), complexify_outer_B(B)


class RealSDMM:
    def compute_product(self, A, B, leakage):
        """Run the procedure"""

        # Compute parameters
        noise_variance_A = 2 * compute_noise_variance(
            self.partitions_A, self.X, self.N, leakage
        )
        noise_variance_B = 2 * compute_noise_variance(
            self.partitions_B, self.X, self.N, leakage
        )

        # Encode matrices
        A_encoded = self.encode_A(A, noise_variance_A)
        B_encoded = self.encode_B(B, noise_variance_B)

        # Compute products
        products_with_indices = [
            (A_encoded[i] @ B_encoded[i], i) for i in range(self.N)
        ]
        shuffle(products_with_indices)

        # Decode the result
        C = self.decode(products_with_indices)

        return C

    def run_simulation(
        self,
        sizes: tuple[int, int, int],
        leakage: float,
        repeat: int | None = None,
        ord="fro",
        dtype=np.float64,
        relative: bool = False,
    ) -> float | np.ndarray[float]:
        """Run the simulation of the scheme over real numbers"""

        if repeat:
            errors = [
                self.run_simulation(sizes, leakage, ord=ord, dtype=dtype)
                for _ in range(repeat)
            ]
            return np.array(errors)

        t, s, r = sizes

        # Choose real matrices
        A = np.random.uniform(-1, 1, size=(t, s))
        B = np.random.uniform(-1, 1, size=(s, r))

        # Do the computation
        C = self.compute_product(A.astype(dtype), B.astype(dtype), leakage)

        # Compute the error
        absolute_error = float(np.linalg.norm(A @ B - C, ord=ord))

        if relative:
            relative_error = absolute_error / float(np.linalg.norm(A @ B, ord=ord))

            return relative_error

        return absolute_error


class RealSDMMInner(RealSDMM):
    def encode_A(self, A, noise_variance):
        """Encode the matrix A"""

        A = complexify_inner_A(A)

        A_partitions = np.split(A, self.partitions_A, axis=1)

        size = A_partitions[0].shape
        dtype = A_partitions[0].dtype
        scale = sqrt(noise_variance)

        random_matrices = [
            circular_normal(scale=scale, size=size).astype(dtype) for _ in range(self.X)
        ]

        A_encoded = []
        evals = np.exp(2j * pi * np.arange(self.N) / self.N)

        for i in range(self.N):
            encoded_matrix = self.polynomial_A(evals[i], A_partitions, random_matrices)
            encoded_matrix = np.block([[encoded_matrix.real, encoded_matrix.imag]])
            A_encoded.append(encoded_matrix)

        return A_encoded

    def encode_B(self, B, noise_variance):
        """Encode the matrix B"""

        B = complexify_inner_B(B)

        B_partitions = np.split(B, self.partitions_B, axis=0)

        size = B_partitions[0].shape
        dtype = B_partitions[0].dtype
        scale = sqrt(noise_variance)

        random_matrices = [
            circular_normal(scale=scale, size=size).astype(dtype) for _ in range(self.X)
        ]

        B_encoded = []
        evals = np.exp(2j * pi * np.arange(self.N) / self.N)

        for i in range(self.N):
            encoded_matrix = self.polynomial_B(evals[i], B_partitions, random_matrices)
            encoded_matrix = np.block([[encoded_matrix.real], [-encoded_matrix.imag]])
            B_encoded.append(encoded_matrix)

        return B_encoded

    def decode(self, products_with_indices):
        """Decode the product from the encoded computations"""

        assert len(products_with_indices) >= self.R

        products_with_indices = products_with_indices[: self.R]
        products, indices = zip(*products_with_indices)

        coefficients = self.decoding_coefficients(indices)

        return sum(
            np.kron(coeff, product) for coeff, product in zip(coefficients, products)
        )


class RealMatDot(RealSDMMInner):
    def __init__(self, M, X, S):
        self.M = M
        self.X = X
        self.S = S
        self.R = 2 * M + 4 * X - 1
        self.N = self.R + S

        self.partitions_A = M
        self.partitions_B = M

    def polynomial_A(self, eval, A_partitions, random_matrices):
        return sum(A_partitions[j] * eval**j for j in range(self.M)) + sum(
            random_matrices[j] * eval ** (self.M + j) for j in range(self.X)
        )

    def polynomial_B(self, eval, B_partitions, random_matrices):
        return sum(B_partitions[j] * eval ** (-j) for j in range(self.M)) + sum(
            random_matrices[j] * eval ** (j + 1) for j in range(self.X)
        )

    def decoding_coefficients(self, indices):
        assert len(indices) == self.R

        evals = np.exp(2j * pi * np.array(indices) / self.N)

        G = np.array(
            [
                [eval ** (j - (self.M + 2 * self.X - 1)) for eval in evals]
                for j in range(self.R)
            ]
        )

        return np.linalg.inv(G)[:, self.M + 2 * self.X - 1].real


class RealDFT(RealSDMMInner):
    def __init__(self, M, X):
        self.M = M
        self.X = X
        self.R = M + 2 * X
        self.N = self.R

        self.partitions_A = M
        self.partitions_B = M

    def polynomial_A(self, eval, A_partitions, random_matrices):
        return sum(A_partitions[j] * eval**j for j in range(self.M)) + sum(
            random_matrices[j] * eval ** (self.M + j) for j in range(self.X)
        )

    def polynomial_B(self, eval, B_partitions, random_matrices):
        return sum(B_partitions[j] * eval ** (-j) for j in range(self.M)) + sum(
            random_matrices[j] * eval ** (j + 1) for j in range(self.X)
        )

    def decoding_coefficients(self, indices):
        assert len(indices) == self.R

        return np.ones(self.N) / self.N


class RealSDMMOuter(RealSDMM):
    def encode_A(self, A, noise_variance):
        """Encode the matrix A"""

        A = complexify_outer_A(A)

        A_partitions = np.split(A, self.partitions_A, axis=0)

        size = A_partitions[0].shape
        dtype = A_partitions[0].dtype
        scale = sqrt(noise_variance)

        random_matrices = [
            circular_normal(scale=scale, size=size).astype(dtype) for _ in range(self.X)
        ]

        A_encoded = []
        evals = np.exp(2j * pi * np.arange(self.N) / self.N)

        for i in range(self.N):
            encoded_matrix = self.polynomial_A(evals[i], A_partitions, random_matrices)
            encoded_matrix = np.block([[encoded_matrix.real], [encoded_matrix.imag]])
            A_encoded.append(encoded_matrix)

        return A_encoded

    def encode_B(self, B, noise_variance):
        """Encode the matrix B"""

        B = complexify_outer_B(B)

        B_partitions = np.split(B, self.partitions_B, axis=1)

        size = B_partitions[0].shape
        dtype = B_partitions[0].dtype
        scale = sqrt(noise_variance)

        random_matrices = [
            circular_normal(scale=scale, size=size).astype(dtype) for _ in range(self.X)
        ]

        B_encoded = []
        evals = np.exp(2j * pi * np.arange(self.N) / self.N)

        for i in range(self.N):
            encoded_matrix = self.polynomial_B(evals[i], B_partitions, random_matrices)
            encoded_matrix = np.block([[encoded_matrix.real, encoded_matrix.imag]])
            B_encoded.append(encoded_matrix)

        return B_encoded

    def decode(self, products_with_indices: list[tuple[np.ndarray, int]]):
        """Decode the product from the encoded computations"""

        assert len(products_with_indices) >= self.R

        products_with_indices = products_with_indices[: self.R]
        products, indices = zip(*products_with_indices)

        products_plus = [(nw(P) - se(P)) + 1j * (ne(P) + sw(P)) for P in products]
        products_minus = [(nw(P) + se(P)) + 1j * (-ne(P) + sw(P)) for P in products]

        coefficients_plus = self.decoding_coefficients_plus(indices)
        coefficients_minus = self.decoding_coefficients_minus(indices)

        result_plus = sum(
            np.kron(coeff, product)
            for coeff, product in zip(coefficients_plus, products_plus)
        )

        result_minus = sum(
            np.kron(coeff, product)
            for coeff, product in zip(coefficients_minus, products_minus)
        )

        return assemble_outer(result_plus, result_minus)


class RealGASP(RealSDMMOuter):
    def __init__(self, K, L, X=0, S=0):
        self.K = K
        self.L = L
        self.X = X
        self.S = S
        self.R = 2 * K * L + K + 3 * X - 2
        self.N = self.R + S
        self.partitions_A = K
        self.partitions_B = L

    def polynomial_A(self, eval, A_partitions, random_matrices):
        return sum(random_matrices[j] * eval**j for j in range(self.X)) + sum(
            A_partitions[j] * eval ** (self.K * self.L + 2 * self.X - 1 + j)
            for j in range(self.K)
        )

    def polynomial_B(self, eval, B_partitions, random_matrices):
        return sum(random_matrices[j] * eval**j for j in range(self.X)) + sum(
            B_partitions[j] * eval ** (self.K + self.X - 1 + self.K * j)
            for j in range(self.L)
        )

    def decoding_coefficients_plus(self, indices):
        assert len(indices) == self.R

        evals = np.exp(2j * pi * np.array(indices) / self.N)

        G = np.array([[eval**j for eval in evals] for j in range(self.R)])

        coeffs = np.zeros((self.R, self.K, self.L), dtype=np.complex128)
        Ginv = np.linalg.inv(G)

        for k in range(self.K):
            for l in range(self.L):
                coeffs[:, k, l] = Ginv[
                    :, self.K * self.L + self.K + 3 * self.X - 2 + k + self.K * l
                ]

        return coeffs

    def decoding_coefficients_minus(self, indices):
        assert len(indices) == self.R

        evals = np.exp(2j * pi * np.array(indices) / self.N)

        offset = -(self.K * self.L + self.X - 1)

        G = np.array([[eval ** (j + offset) for eval in evals] for j in range(self.R)])

        coeffs = np.zeros((self.R, self.K, self.L), dtype=np.complex128)
        Ginv = np.linalg.inv(G)

        for k in range(self.K):
            for l in range(self.L):
                coeffs[:, k, l] = Ginv[
                    :, self.K * (self.L - 1) + self.X + k - self.K * l - offset
                ]

        return coeffs


class RealA3S(RealSDMMOuter):
    def __init__(self, K, L, X=0, S=0):
        self.K = K
        self.L = L
        self.X = X
        self.S = S
        self.R = (K + X) * (L + 1) + X - 1
        self.N = self.R + S
        self.partitions_A = K
        self.partitions_B = L

    def polynomial_A(self, eval, A_partitions, random_matrices):
        return sum(A_partitions[j] * eval**j for j in range(self.K)) + sum(
            random_matrices[j] * eval ** (self.K + j) for j in range(self.X)
        )

    def polynomial_B(self, eval, B_partitions, random_matrices):
        return sum(
            B_partitions[j] * eval ** ((self.K + self.X) * j) for j in range(self.L)
        ) + sum(
            random_matrices[j] * eval ** ((self.K + self.X) * (self.L) + j)
            for j in range(self.X)
        )

    def decoding_coefficients_plus(self, indices):
        assert len(indices) == self.R

        evals = np.exp(2j * pi * np.array(indices) / self.N)

        G = np.array([[eval**j for eval in evals] for j in range(self.R)])

        coeffs = np.zeros((self.R, self.K, self.L), dtype=np.complex128)
        Ginv = np.linalg.inv(G)

        for k in range(self.K):
            for l in range(self.L):
                coeffs[:, k, l] = Ginv[:, k + (self.K + self.X) * l]

        return coeffs

    def decoding_coefficients_minus(self, indices):
        assert len(indices) == self.R

        evals = np.exp(2j * pi * np.array(indices) / self.N)

        offset = -((self.K + self.X) * self.L + self.X - 1)

        G = np.array([[eval ** (j + offset) for eval in evals] for j in range(self.R)])

        coeffs = np.zeros((self.R, self.K, self.L), dtype=np.complex128)
        Ginv = np.linalg.inv(G)

        for k in range(self.K):
            for l in range(self.L):
                coeffs[:, k, l] = Ginv[:, k - (self.K + self.X) * l - offset]

        return coeffs


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    M = 10
    K = 4
    L = 4
    X = 2
    S = 3

    sizes = (120, 120, 120)
    leakage = 1.0

    matdot = RealMatDot(M, X, S)
    matdot_error = matdot.run_simulation(sizes, leakage)

    dft = RealDFT(60, X)
    dft_error = dft.run_simulation(sizes, leakage)

    gasp = RealGASP(K, L, X, S)
    gasp_error = gasp.run_simulation(sizes, leakage)

    a3s = RealA3S(K, L, X, S)
    a3s_error = a3s.run_simulation(sizes, leakage)

    print(
        f"MatDot: N = {matdot.N:3d}, M = {matdot.M:3d}, X = {matdot.X:3d}, S = {matdot.S:3d}, error = {matdot_error}"
    )
    print(
        f"DFT:    N = {dft.N:3d}, M = {dft.M:3d}, X = {dft.X:3d}, S =   0, error = {dft_error}"
    )
    print(
        f"GASP:   N = {gasp.N:3d}, K = {gasp.K:3d}, L = {gasp.L:3d}, X = {matdot.X:3d}, S = {matdot.S:3d}, error = {gasp_error}"
    )
    print(
        f"A3S:    N = {a3s.N:3d}, K = {a3s.K:3d}, L = {a3s.L:3d}, X = {a3s.X:3d}, S = {a3s.S:3d}, error = {a3s_error}"
    )
