from math import factorial, sqrt
from random import shuffle

import numpy as np


def circular_normal(loc: complex = 0.0j, scale: float = 1.0, size=None):
    """Create an array whose elements are chosen from a circularly symmetric normal distribution"""

    real = np.random.normal(loc=loc.real, scale=scale / sqrt(2), size=size)
    imag = np.random.normal(loc=loc.imag, scale=scale / sqrt(2), size=size)

    return real + 1j * imag


def unit_disk_uniform(size):
    """Create an array whose elements are complex numbers drawn uniformly from the unit disk

    >>> bool((np.abs(unit_disk_uniform(size=(100,))) <= 1).all())
    True
    >>> bool(abs(unit_disk_uniform(size=(10000,)).mean()) < 0.05)
    True
    """

    r = np.sqrt(np.random.uniform(size=size))
    t = 2 * np.pi * np.random.uniform(size=size)

    return r * np.exp(1j * t)


def Pi(n: int):
    """Compute an entry of the sequence https://oeis.org/A010551

    >>> [Pi(n) for n in range(10)]
    [1, 1, 1, 2, 4, 12, 36, 144, 576, 2880]
    """

    k = n // 2

    if n % 2 == 0:
        return factorial(k) ** 2

    return factorial(k) ** 2 * (k + 1)


def compute_noise_variance(P: int, X: int, N: int, delta: float, optimize_X1=True):
    """Compute the required noise variance to achieve a leakage of less than delta

    >>> compute_noise_variance(2, 1, 10, 0.1, optimize_X1=False)
    20.0
    """

    if X < 1:
        return 0.0

    if X == 1 and optimize_X1:
        return float(P**2 / np.expm1(P * delta))

    return P * X**3 * N ** (2 * X - 2) / (delta * 4 ** (X - 1) * Pi(X - 1) ** 2)


class ComplexSDMM:
    """Methods for SDMM schemes over complex numbers"""

    def encode_A(self, A, noise_variance):
        """Encode the matrix A"""

        A = A + 0j

        if self.partition == "outer":
            A_partitions = np.split(A, self.partitions_A, axis=0)
        elif self.partition == "inner":
            A_partitions = np.split(A, self.partitions_A, axis=1)
        else:
            raise Exception("incorrect partition")

        size = A_partitions[0].shape
        dtype = A_partitions[0].dtype
        scale = sqrt(noise_variance)

        random_matrices = [
            circular_normal(scale=scale, size=size).astype(dtype) for _ in range(self.X)
        ]

        A_encoded = []
        theta = 2 * np.pi / self.N
        evals = np.exp(1j * theta * np.arange(self.N))

        for i in range(self.N):
            encoded_matrix = self.polynomial_A(evals[i], A_partitions, random_matrices)
            A_encoded.append(encoded_matrix)

        return A_encoded

    def encode_B(self, B, noise_variance):
        """Encode the matrix B"""

        B = B + 0j

        if self.partition == "outer":
            B_partitions = np.split(B, self.partitions_B, axis=1)
        elif self.partition == "inner":
            B_partitions = np.split(B, self.partitions_B, axis=0)
        else:
            raise Exception("incorrect partition")

        size = B_partitions[0].shape
        dtype = B_partitions[0].dtype
        scale = sqrt(noise_variance)

        random_matrices = [
            circular_normal(scale=scale, size=size).astype(dtype) for _ in range(self.X)
        ]

        B_encoded = []
        theta = 2 * np.pi / self.N
        evals = np.exp(1j * theta * np.arange(self.N))

        for i in range(self.N):
            encoded_matrix = self.polynomial_B(evals[i], B_partitions, random_matrices)
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

    def compute_product(self, A, B, leakage):
        """Run the procedure"""

        # Compute parameters
        noise_variance_A = compute_noise_variance(
            self.partitions_A, self.X, self.N, leakage
        )
        noise_variance_B = compute_noise_variance(
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
        sizes,
        leakage,
        repeat=None,
        ord="fro",
        dtype=np.complex128,
        relative: bool = False,
    ) -> float | np.ndarray[float]:
        """Run the simulation of the scheme over complex numbers"""

        if repeat:
            errors = [
                self.run_simulation(
                    sizes, leakage, ord=ord, dtype=dtype, relative=relative
                )
                for _ in range(repeat)
            ]
            return np.array(errors)

        t, s, r = sizes

        # Choose complex matrices
        A = unit_disk_uniform(size=(t, s))
        B = unit_disk_uniform(size=(s, r))

        # Do the computation
        C = self.compute_product(A.astype(dtype), B.astype(dtype), leakage)

        # Compute the error
        absolute_error = float(np.linalg.norm(A @ B - C, ord=ord))

        if relative:
            relative_error = absolute_error / float(np.linalg.norm(A @ B, ord=ord))

            return relative_error

        return absolute_error

    def run_simulation_real(
        self,
        sizes,
        leakage,
        repeat=None,
        ord="fro",
        dtype=np.complex128,
        relative: bool = False,
    ) -> float | np.ndarray[float]:
        """Run the simulation of the scheme over real numbers"""

        if repeat:
            errors = [
                self.run_simulation(
                    sizes, leakage, ord=ord, dtype=dtype, relative=relative
                )
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


class ComplexDFT(ComplexSDMM):
    partition = "inner"

    def __init__(self, M, X):
        self.M = M
        self.X = X
        self.N = M + 2 * X
        self.R = self.N
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
        return np.ones(self.N) / self.N


class ComplexMatDot(ComplexSDMM):
    partition = "inner"

    def __init__(self, M, X, S):
        self.M = M
        self.X = X
        self.R = 2 * M + 2 * X - 1
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

        theta = 2 * np.pi / self.N
        evals = np.exp(1j * theta * np.arange(self.N))

        G = np.array(
            [[evals[i] ** (j - self.M + 1) for i in indices] for j in range(self.R)]
        )

        return np.linalg.inv(G)[:, self.M - 1]


class ComplexA3S(ComplexSDMM):
    partition = "outer"

    def __init__(self, K, L, X, S):
        self.K = K
        self.L = L
        self.X = X
        self.R = (K + X) * (L + 1) - 1
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
            random_matrices[j] * eval ** ((self.K + self.X) * (self.L - 1) + self.K + j)
            for j in range(self.X)
        )

    def decoding_coefficients(self, indices):
        assert len(indices) == self.R

        theta = 2 * np.pi / self.N
        evals = np.exp(1j * theta * np.arange(self.N))

        G = np.array([[evals[i] ** j for i in indices] for j in range(self.R)])

        coeffs = np.zeros((self.R, self.K, self.L), dtype=np.complex128)
        Ginv = np.linalg.inv(G)

        for k in range(self.K):
            for l in range(self.L):
                coeffs[:, k, l] = Ginv[:, (self.K + self.X) * l + k]

        return coeffs


class ComplexGASP(ComplexSDMM):
    partition = "outer"

    def __init__(self, K, L, X, S):
        self.K = K
        self.L = L
        self.X = X
        self.R = 2 * K * L + 2 * X - 1
        self.N = self.R + S
        self.partitions_A = K
        self.partitions_B = L

    def polynomial_A(self, eval, A_partitions, random_matrices):
        return sum(random_matrices[j] * eval**j for j in range(self.X)) + sum(
            A_partitions[j] * eval ** (self.K * (self.L - 1) + self.X + j)
            for j in range(self.K)
        )

    def polynomial_B(self, eval, B_partitions, random_matrices):
        return sum(random_matrices[j] * eval**j for j in range(self.X)) + sum(
            B_partitions[j] * eval ** (self.K + self.X - 1 + self.K * j)
            for j in range(self.L)
        )

    def decoding_coefficients(self, indices):
        assert len(indices) == self.R

        theta = 2 * np.pi / self.N
        evals = np.exp(1j * theta * np.arange(self.N))

        G = np.array([[evals[i] ** j for i in indices] for j in range(self.R)])

        coeffs = np.zeros((self.R, self.K, self.L), dtype=np.complex128)
        Ginv = np.linalg.inv(G)

        for k in range(self.K):
            for l in range(self.L):
                coeffs[:, k, l] = Ginv[
                    :, self.K * self.L + 2 * self.X - 1 + k + self.K * l
                ]

        return coeffs


if __name__ == "__main__":
    import doctest

    doctest.testmod()
