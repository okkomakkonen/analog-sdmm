import numpy as np
import pandas as pd

from complex_sdmm import ComplexDFT, ComplexMatDot, ComplexA3S, ComplexGASP
from real_sdmm import RealDFT, RealMatDot, RealA3S, RealGASP

REPEAT = 400
CONFIDENCE = 0.9
ORD = "fro"
COMPLEX_DTYPE = np.csingle
REAL_DTYPE = np.single
RELATIVE = True


def save_data(xs, ys, name, confidence=CONFIDENCE):
    """Save the data into .dat files consisting of the columns x, y, ey-, ey+"""
    lower = (1 - confidence) / 2
    upper = 1 - lower

    data = np.array(
        [
            (
                x,
                np.median(y),
                np.median(y) - np.quantile(y, lower),
                np.quantile(y, upper) - np.median(y),
            )
            for x, y in zip(xs, ys)
        ]
    )

    df = pd.DataFrame(data, columns=["x", "y", "ey-", "ey+"])

    df.to_csv(f"data/{name}.dat", sep="\t", index=False)


# A: plot accuracy as a function of leakage, complex inner schemes

sizes = (256, 256, 256)
M = 8
X = 3
Ss = [0, 2, 4]
leakages = 10 ** np.linspace(-4.5, 1.5, 8)

complex_dft_errors_leakage = [
    ComplexDFT(M=M, X=X).run_simulation(
        sizes, leakage, repeat=REPEAT, ord=ORD, dtype=COMPLEX_DTYPE, relative=RELATIVE
    )
    for leakage in leakages
]

save_data(leakages, complex_dft_errors_leakage, "complex_dft_errors_leakage")

for S in Ss:
    complex_matdot_errors_leakage = [
        ComplexMatDot(M=M, X=X, S=S).run_simulation(
            sizes,
            leakage,
            repeat=REPEAT,
            ord=ORD,
            dtype=COMPLEX_DTYPE,
            relative=RELATIVE,
        )
        for leakage in leakages
    ]

    save_data(
        leakages, complex_matdot_errors_leakage, f"complex_matdot_errors_leakage_{S}"
    )


# B: plot accuracy as a function of leakage, complex outer schemes

sizes = (256, 256, 256)
K = 4
L = 4
X = 3
S = 4
leakages = 10 ** np.linspace(-4.5, 1.5, 8)

complex_a3s_errors_leakage = [
    ComplexA3S(K, L, X, S).run_simulation(
        sizes, leakage, repeat=REPEAT, ord=ORD, dtype=COMPLEX_DTYPE, relative=RELATIVE
    )
    for leakage in leakages
]

save_data(leakages, complex_a3s_errors_leakage, "complex_a3s_errors_leakage")

complex_gasp_errors_leakage = [
    ComplexGASP(K, L, X, S).run_simulation(
        sizes, leakage, repeat=REPEAT, ord=ORD, dtype=COMPLEX_DTYPE, relative=RELATIVE
    )
    for leakage in leakages
]

save_data(leakages, complex_gasp_errors_leakage, "complex_gasp_errors_leakage")

# C: plot accuracy as a function of security parameter, complex inner parameters

sizes = (256, 256, 256)
M = 8
Xs = [1, 2, 3, 4, 5, 6, 7, 8]
Ss = [0, 2, 4]
leakage = 1e-1

complex_dft_errors_security = [
    ComplexDFT(M=M, X=X).run_simulation(
        sizes, leakage, repeat=REPEAT, ord=ORD, dtype=COMPLEX_DTYPE, relative=RELATIVE
    )
    for X in Xs
]

save_data(Xs, complex_dft_errors_security, "complex_dft_errors_security")

for S in Ss:
    complex_matdot_errors_security = [
        ComplexMatDot(M=M, X=X, S=S).run_simulation(
            sizes,
            leakage,
            repeat=REPEAT,
            ord=ORD,
            dtype=COMPLEX_DTYPE,
            relative=RELATIVE,
        )
        for X in Xs
    ]

    save_data(Xs, complex_matdot_errors_security, f"complex_matdot_errors_security_{S}")

# D: plot accuracy as a function of leakage, real inner schemes

sizes = (256, 256, 256)
M = 8
X = 3
Ss = [0, 2, 4]

leakages = 10 ** np.linspace(-3, 1.5, 8)

real_dft_errors_leakage = [
    RealDFT(M=M, X=X).run_simulation(
        sizes, leakage, repeat=REPEAT, ord=ORD, dtype=REAL_DTYPE, relative=RELATIVE
    )
    for leakage in leakages
]

save_data(leakages, real_dft_errors_leakage, "real_dft_errors_leakage")

for S in Ss:
    real_matdot_errors_leakage = [
        RealMatDot(M=M, X=X, S=S).run_simulation(
            sizes, leakage, repeat=REPEAT, ord=ORD, dtype=REAL_DTYPE, relative=RELATIVE
        )
        for leakage in leakages
    ]

    save_data(leakages, real_matdot_errors_leakage, f"real_matdot_errors_leakage_{S}")

# E: plot accuracy as a function of leakage, real outer schemes

sizes = (256, 256, 256)
K = 4
L = 4
X = 3
S = 4
leakages = 10 ** np.linspace(-3, 1.5, 8)

real_a3s_errors_leakage = [
    RealA3S(K, L, X, S).run_simulation(
        sizes, leakage, repeat=REPEAT, ord=ORD, dtype=REAL_DTYPE, relative=RELATIVE
    )
    for leakage in leakages
]

save_data(leakages, real_a3s_errors_leakage, "real_a3s_errors_leakage")

real_gasp_errors_leakage = [
    RealGASP(K, L, X, S).run_simulation(
        sizes, leakage, repeat=REPEAT, ord=ORD, dtype=REAL_DTYPE, relative=RELATIVE
    )
    for leakage in leakages
]

save_data(leakages, real_gasp_errors_leakage, "real_gasp_errors_leakage")
