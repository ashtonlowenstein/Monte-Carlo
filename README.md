# Monte-Carlo
Practice with Monte Carlo algorithms

These are some scripts I wrote while learning the basics of Monte Carlo methods. I started with the Ising model, then applied what I learned to generate random matrices in the Gaussian Unitary Ensemble (GUE).

A practical issue in random matrix theory (RMT) is that there is no pre-built way to generate random matrices for Wigner-Dyson distributions more complicated than the base Gaussian case. The Gaussian case has the simplifying feature that each element of the random matrix is independentally normally distributed, and therefore can be easily generated with a package like NumPy. If the joint probability density is more complicated, say $e^{-tr(M^4)}$, then the the matrix elements are coupled together and their joint distribution is a quartic polynomial in $N(N-1)/2$ complex random variables (for an $N \times N$ hermitian matrix).

So, I decided to try to generate matrices with the joint probability density $e^{-tr(M^4)}$, both as a challenge and for the practical purpose of being able to visualize the analog of the Wigner semi-circle for the distribution of the eigenvalues of the matrices. This is along the lines of the 'numerical experiments' performed on the GUE by my advisor Clifford Johnson in some of his work. I did something similar for the GOE in my dissertation.
