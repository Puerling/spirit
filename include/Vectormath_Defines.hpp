#pragma once

#include <Managed_Allocator.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <complex>
#include <vector>

using scalar = float;

// Dynamic Eigen typedefs
using VectorX    = Eigen::Matrix<scalar, -1, 1>;
using RowVectorX = Eigen::Matrix<scalar, 1, -1>;
using MatrixX    = Eigen::Matrix<scalar, -1, -1>;
using SpMatrixX  = Eigen::SparseMatrix<scalar>;

// 3D Eigen typedefs
using Vector3    = Eigen::Matrix<scalar, 3, 1>;
using RowVector3 = Eigen::Matrix<scalar, 1, 3>;
using Matrix3    = Eigen::Matrix<scalar, 3, 3>;

using Vector3c = Eigen::Matrix<std::complex<scalar>, 3, 1>;
using Matrix3c = Eigen::Matrix<std::complex<scalar>, 3, 3>;

// 2D Eigen typedefs
using Vector2 = Eigen::Matrix<scalar, 2, 1>;

// The general field, using the managed allocator
template<typename T>
using field = std::vector<T, managed_allocator<T>>;

struct Site
{
    // Basis index
    int i;
    // Translations of the basis cell
    int translations[3];
};
struct Pair
{
    // Basis indices of first and second atom of pair
    int i, j;
    // Translations of the basis cell of second atom of pair
    int translations[3];
};
struct Triplet
{
    int i, j, k;
    int d_j[3], d_k[3];
};
struct Quadruplet
{
    int i, j, k, l;
    int d_j[3], d_k[3], d_l[3];
};


struct PolynomialBasis {
    Vector3 k1, k2, k3;
};

struct PolynomialTerm
{
    scalar coefficient;
    unsigned int n1, n2, n3;
};

struct AnisotropyPolynomial
{
    Vector3 k1, k2, k3;
    field<PolynomialTerm> terms;
};

struct PolynomialField {
    field<PolynomialBasis> basis;
    field<unsigned int> site_p;
    field<PolynomialTerm> terms;
};

struct Neighbour : Pair
{
    // Shell index
    int idx_shell;
};

// Important fields
using intfield    = field<int>;
using scalarfield = field<scalar>;
using vectorfield = field<Vector3>;

// Additional fields
using pairfield       = field<Pair>;
using tripletfield    = field<Triplet>;
using quadrupletfield = field<Quadruplet>;
using neighbourfield  = field<Neighbour>;
using vector2field    = field<Vector2>;
