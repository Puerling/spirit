#include <Indexing.hpp>
#include <Interaction.hpp>
#include <Vectormath_Defines.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

using Indexing::cu_check_atom_type;

__global__ void CU_E_Biaxial_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * indices, const PolynomialBasis * bases, const unsigned int * site_p, const PolynomialTerm * terms,
    scalar * energy, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
            {
                const scalar s1 = bases[iani].k1.dot( spins[ispin] );
                const scalar s2 = bases[iani].k2.dot( spins[ispin] );
                const scalar s3 = bases[iani].k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                scalar result = 0;
                for( int iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
                {
                    const auto & [coeff, n1, n2, n3] = terms[iterm];
                    result += coeff * pow( sin_theta_2, n1 ) * pow( s2, n2 ) * pow( s3, n3 );
                }
                energy[ispin] += result;
            }
        }
    }
}

void Interaction::Energy_per_Spin( const Geometry & geometry, const vectorfield & spins, scalarfield & energy )
{
    const int size = geometry.n_cells_total;
    CU_E_Biaxial_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry.atom_types.data(), geometry.n_cell_atoms, this->indices.size(), this->indices.data(),
        this->bases.data(), this->site_p.data(), this->terms.data(), energy.data(), size );
    CU_CHECK_AND_SYNC();
}

__global__ void CU_Gradient_Biaxial_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * indices, const PolynomialBasis * bases, const unsigned int * site_p, const PolynomialTerm * terms,
    Vector3 * gradient, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
            {
                Vector3 result = Vector3::Zero();

                const auto & [k1, k2, k3] = bases[iani];

                const scalar s1 = k1.dot( spins[ispin] );
                const scalar s2 = k2.dot( spins[ispin] );
                const scalar s3 = k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
                {
                    const auto & [coeff, n1, n2, n3] = terms[iterm];

                    const scalar a = pow( s2, n2 );
                    const scalar b = pow( s3, n3 );
                    const scalar c = pow( sin_theta_2, n1 );

                    if( n1 > 0 )
                        result += k1 * ( coeff * a * b * n1 * ( -2.0 * s1 * pow( sin_theta_2, n1 - 1 ) ) );
                    if( n2 > 0 )
                        result += k2 * ( coeff * b * c * n2 * pow( s2, n2 - 1 ) );
                    if( n3 > 0 )
                        result += k3 * ( coeff * a * c * n3 * pow( s3, n3 - 1 ) );
                }

                gradient[ispin] += result;
            }
        }
    }
}

void Interaction::Gradient( const Geometry & geometry, const vectorfield & spins, vectorfield & gradient )
{
    const int size = geometry.n_cells_total;
    CU_Gradient_Biaxial_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry.atom_types.data(), geometry.n_cell_atoms, this->indices.size(), this->indices.data(),
        this->bases.data(), this->site_p.data(), this->terms.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
};
