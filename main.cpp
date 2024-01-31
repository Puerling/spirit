#include "Geometry.hpp"
#include "Interaction.hpp"
#include "Vectormath.hpp"
#include "Vectormath_Defines.hpp"

#include <fmt/core.h>
#include <iostream>
#include <random>

#define INFO( arg ) ( std::cout << arg << "\n" );

namespace C {
double constexpr Pi = 3.141592653589793238462643383279502884197169399375105820974;
}

namespace
{

template<typename... Args>
Interaction make_interaction( Args &&... args )
{
    static_assert( std::is_constructible_v<Interaction, Args...> );
    return Interaction( std::forward<Args>( args )... );
};

Vector3 make_spherical( const scalar theta, const scalar phi )
{
    return { cos( phi ) * sin( theta ), sin( phi ) * sin( theta ), cos( theta ) };
};

} // namespace

int main( int argc, const char * argv[] )
{
    static const intfield init_indices{0};
    static const field<PolynomialBasis> init_bases { PolynomialBasis{
        Vector3{ 0, 0, 1 },
        Vector3{ 1, 0, 0 },
        Vector3{ 0, 1, 0 },
    }
    };

    const Geometry geometry( intfield{ 10, 10, 1 }, 1 );
    Interaction interaction{};

    auto term_info = []( const auto &... terms )
    {
        return fmt::format( "{} term(s):\n", sizeof...( terms ) )
               + ( ...
                   + fmt::format(
                       "    n1={}, n2={}, n3={}, c={}\n", terms.n1, terms.n2, terms.n3, terms.coefficient ) );
    };

    const auto test = [&geometry, &interaction,
                       &term_info]( const int idx, const scalar theta, const scalar phi, const auto &... terms )
    {
        interaction.setParameters( init_indices, init_bases, {0u, sizeof...(terms)}, field<PolynomialTerm>{terms...} );

        vectorfield spins( geometry.nos, make_spherical( theta, phi ) );
        scalarfield energy( geometry.nos, 0.0 );
        interaction.Energy_per_Spin( geometry, spins, energy );

        // reference energy
        const scalar ref_energy
            = ( ...
                + ( terms.coefficient * pow( sin( theta ), 2 * terms.n1 + terms.n2 + terms.n3 )
                    * pow( cos( phi ), terms.n2 ) * pow( sin( phi ), terms.n3 ) ) );

        for( const auto & e : energy )
        {
            INFO( "Energy:" )
            INFO( "trial: " << idx << ", theta=" << theta << ", phi=" << phi );
            INFO( term_info( terms... ) );
        };

        INFO( "Total Energy:" )
        INFO( "trial: " << idx << ", theta=" << theta << ", phi=" << phi );
        INFO( term_info( terms... ) );

        vectorfield gradient( geometry.nos, Vector3::Zero() );
        interaction.Gradient( geometry, spins, gradient );

        const auto k1 = init_bases[0].k1;
        const auto k2 = init_bases[0].k2;
        const auto k3 = init_bases[0].k3;

        Vector3 ref_gradient
            = ( ... +
                [s1 = cos( theta ), s2 = sin( theta ) * cos( phi ), s3 = sin( theta ) * sin( phi ), &k1, &k2,
                 &k3]( const auto & term )
                {
                    Vector3 result{ 0, 0, 0 };
                    const scalar a = pow( s2, term.n2 );
                    const scalar b = pow( s3, term.n3 );
                    const scalar c = pow( 1 - s1 * s1, term.n1 );

                    const auto & [coefficient, n1, n2, n3] = term;
                    if( n1 > 0 )
                        result += k1 * ( coefficient * a * b * n1 * ( -2 * s1 * pow( 1 - s1 * s1, n1 - 1 ) ) );
                    if( n2 > 0 )
                        result += k2 * ( coefficient * b * c * n2 * pow( s2, n2 - 1 ) );
                    if( n3 > 0 )
                        result += k3 * ( coefficient * a * c * n3 * pow( s3, n3 - 1 ) );
                    return result;
                }( terms ) );

        for( const auto & g : gradient )
        {
            for( std::size_t i = 0; i < 3; ++i )
            {
                INFO( "trial: " << idx << ", theta=" << theta << ", phi=" << phi );
                INFO( term_info( terms... ) );
                INFO( "Gradient(expected): " << ref_gradient.transpose() );
                INFO( "Gradient(actual):   " << g.transpose() );
            }
        }
    };

    auto rng         = std::mt19937( 3548368 );
    auto angle_theta = std::uniform_real_distribution<scalar>( 0, C::Pi );
    auto angle_phi   = std::uniform_real_distribution<scalar>( -2 * C::Pi, 2 * C::Pi );
    auto coeff       = std::uniform_real_distribution<scalar>( -10.0, 10.0 );
    auto exp         = std::uniform_int_distribution<unsigned int>( 0, 6 );

    for( int n = 0; n < 6; ++n )
    {
        const scalar theta = angle_theta( rng );
        const scalar phi   = angle_phi( rng );

        std::array terms{
            PolynomialTerm{ coeff( rng ), exp( rng ), exp( rng ), exp( rng ) },
            PolynomialTerm{ coeff( rng ), exp( rng ), exp( rng ), exp( rng ) },
        };

        for( const auto & term : terms )
            test( n, theta, phi, term );

        for( const auto & term_a : terms )
            for( const auto & term_b : terms )
                test( n, theta, phi, term_a, term_b );
    }
}
