#pragma once

#include <Vectormath_Defines.hpp>
#include <Geometry.hpp>

#include <cassert>

class Interaction
{
public:
    Interaction(
        intfield pIndices, field<PolynomialBasis> pBases, field<unsigned int> pSite_ptr,
        field<PolynomialTerm> pTerms ) noexcept
        : indices( std::move( pIndices ) ),
          bases( std::move( pBases ) ),
          site_p( std::move( pSite_ptr ) ),
          terms( std::move( pTerms ) )
{
};

    Interaction() noexcept = default;

    void setParameters(
        const intfield & pIndices, const field<PolynomialBasis> & pBases, const field<unsigned int> & pSite_ptr,
        const field<PolynomialTerm> & pTerms )
    {
        assert( pIndices.size() == pBases.size() );
        assert( ( pIndices.empty() && pSite_ptr.empty() ) || ( pIndices.size() + 1 == pSite_ptr.size() ) );
        assert( pSite_ptr.empty() || pSite_ptr.back() == pTerms.size() );

        this->indices = pIndices;
        this->bases   = pBases;
        this->site_p  = pSite_ptr;
        this->terms   = pTerms;
    };
    void getParameters(
        intfield & pIndices, field<PolynomialBasis> & pBases, field<unsigned int> & pSite_ptr,
        field<PolynomialTerm> & pTerms ) const
    {
        pIndices  = this->indices;
        pBases    = this->bases;
        pSite_ptr = this->site_p;
        pTerms    = this->terms;
    };

    [[nodiscard]] std::size_t getN_Atoms() const
    {
        return indices.size();
    }

    [[nodiscard]] std::size_t getN_Terms() const
    {
        return terms.size();
    }

    void Energy_per_Spin( const Geometry & geometry, const vectorfield & spins, scalarfield & energy );

    void Gradient( const Geometry & geometry, const vectorfield & spins, vectorfield & gradient );

private:
    intfield indices;
    field<PolynomialBasis> bases;
    field<unsigned int> site_p;
    field<PolynomialTerm> terms;
};
