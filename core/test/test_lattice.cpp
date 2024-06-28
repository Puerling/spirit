#ifdef SPIRIT_ENABLE_LATTICE
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <data/State.hpp>
#include <engine/spin_lattice/Hamiltonian.hpp>
#include <engine/spin_lattice/Method_LLG.hpp>
#include <io/Configparser.hpp>
#include <utility/Formatters_Eigen.hpp>

#include "catch.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fmt/format.h>

using Catch::Matchers::WithinAbs;

// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-10;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-12;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-12;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-6;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-7;
#else
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-2;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-3;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-4;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-5;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-6;
#endif

Data::Geometry make_dimer_geometry()
{
    return Data::Geometry(
        Data::Geometry::BravaisVectorsSC(), /*n_cells=*/{ 2, 1, 1 }, /*cell_atoms=*/{ Vector3::Zero() },
        Data::Basis_Cell_Composition{ /*disordered=*/false, /*iatom=*/{ 0 }, /*atom_type=*/{ 0 }, /*mu_s=*/{ 1 },
                                      /*concentration=*/{} },
        /*lattice_constant=*/1.0, Data::Pinning(), Data::Defects() );
}

TEST_CASE( "Dimer simulation through the (spring) lattice Hamiltonian", "[lattice]" )
{
    using namespace Engine::SpinLattice;
    using LatticeHamiltonian = Hamiltonian<
        StateType, Interaction::StandaloneAdaptor<StateType>, Interaction::Lattice_Kinetic,
        Interaction::Lattice_Spring_Potential>;

    auto hamiltonian = HamiltonianVariant( LatticeHamiltonian(
        /*geometry=*/make_dimer_geometry(),
        /*bounary_conditions=*/intfield{ 0, 0, 0 },
        typename Interaction::Lattice_Kinetic::Data(
            /*indices=*/{ 0 }, /*magnitudes=*/{ 1.0, 1.0, 1.0 },
            /*normals=*/{ Vector3{ 1, 0, 0 }, Vector3{ 0, 1, 0 }, Vector3{ 0, 0, 1 } } ),
        typename Interaction::Lattice_Spring_Potential::Data(
            /*pairs=*/pairfield{ Pair{ 0, 0, { 1, 0, 0 } } }, /*magnitudes=*/scalarfield{ 1.0 } ) ) );

    const auto nos = hamiltonian.get_geometry().nos;

    static constexpr auto labels = make_quantity<std::string_view>( "spin", "displacement", "momentum" );

    SECTION( "Energy and Gradients" )
    {
        static constexpr auto sign = []( const auto idx ) -> scalar { return ( idx % 2 ) == 0 ? -1.0 : 1.0; };
        const auto check_gradient
            = [nos]( const quantity<vectorfield> & gradient, const quantity<Vector3> & gradient_exp )
        {
            for( int i = 0; i < nos; ++i )
            {
                {
                    INFO( fmt::format( "{} gradient, i={}", labels.spin, i ) );
                    INFO( fmt::format( "(found)    = {}", gradient.spin[i].transpose() ) );
                    INFO( fmt::format( "(expected) = {}", gradient_exp.spin.transpose() ) );
                    REQUIRE( gradient.spin[i].isApprox( gradient_exp.spin, epsilon_2 ) );
                }
                {
                    INFO( fmt::format( "{} gradient, i={}", labels.displacement, i ) );
                    INFO( fmt::format( "(found)    = {}", gradient.displacement[i].transpose() ) );
                    INFO( fmt::format( "(expected) = {}", sign( i ) * gradient_exp.displacement.transpose() ) );
                    REQUIRE( gradient.displacement[i].isApprox( sign( i ) * gradient_exp.displacement, epsilon_2 ) );
                }
                {
                    INFO( fmt::format( "{} gradient, i={}", labels.momentum, i ) );
                    INFO( fmt::format( "(found)    = {}", gradient.momentum[i].transpose() ) );
                    INFO( fmt::format( "(expected) = {}", gradient_exp.momentum.transpose() ) );
                    REQUIRE( gradient.momentum[i].isApprox( gradient_exp.momentum, epsilon_2 ) );
                }
            }
        };

        const auto check_energy = [nos]( const scalarfield & energy_per_spin, const scalar eps_expected )
        {
            for( int i = 0; i < nos; ++i )
            {
                INFO( fmt::format( "Energy per spin, i={}", i ) );
                INFO( fmt::format( "(found)    = {}", energy_per_spin[i] ) );
                INFO( fmt::format( "(expected) = {}", eps_expected ) );
                REQUIRE_THAT( energy_per_spin[i], WithinAbs( eps_expected, epsilon_2 ) );
            }
        };

        auto state           = make_quantity( vectorfield{} );
        auto gradient        = make_quantity( vectorfield( nos, Vector3::Zero() ) );
        auto energy_per_spin = scalarfield( nos, 0.0 );

        SECTION( "test case 1" )
        {
            const quantity<Vector3> gradient_exp{ Vector3::Zero(), { 1.0, 0.0, 0.0 }, { 0.5, 0.0, 0.0 } };
            constexpr scalar eps_expected = /*spin*/ 0.0 + /*displacement*/ 0.25 * ( 2.0 - 1.0 ) * ( 2.0 - 1.0 )
                                            + /*momentum*/ 0.5 * ( 0.5 * 0.5 );

            state.spin         = vectorfield( 2, Vector3::Zero() );
            state.displacement = { -0.5 * gradient_exp.displacement, 0.5 * gradient_exp.displacement };
            state.momentum     = { gradient_exp.momentum, gradient_exp.momentum };

            hamiltonian.Energy_per_Spin( state, energy_per_spin );
            check_energy( energy_per_spin, eps_expected );

            hamiltonian.Gradient( state, gradient );
            check_gradient( gradient, gradient_exp );
        };

        SECTION( "test case 2" )
        {
            const quantity<Vector3> gradient_exp{ Vector3::Zero(), Vector3::Zero(), { 0.0, 1.0, 0.0 } };
            constexpr scalar eps_expected = /*spin*/ 0.0 + /*displacement*/ 0.0 + /*momentum*/ 0.5 * ( 1.0 * 1.0 );

            state.spin         = vectorfield{ { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } };
            state.displacement = { Vector3{ 0, 0.5, 0 }, Vector3{ 0, 0.5, 0 } };
            state.momentum     = { gradient_exp.momentum, gradient_exp.momentum };

            hamiltonian.Gradient( state, gradient );
            check_gradient( gradient, gradient_exp );

            hamiltonian.Energy_per_Spin( state, energy_per_spin );
            check_energy( energy_per_spin, eps_expected );
        };

        SECTION( "test case 3" )
        {
            const scalar d = sqrt( 2.0 ) - 1.0;
            const quantity<Vector3> gradient_exp{ Vector3::Zero(), Vector3{ d / sqrt( 2.0 ), d / sqrt( 2.0 ), 0.0 },
                                                  Vector3{ 0.0, 0.0, 1.0 } };
            const scalar eps_expected = /*spin*/ 0.0 + /*displacement*/ 0.25 * d * d + /*momentum*/ 0.5 * ( 1.0 * 1.0 );

            state.spin         = vectorfield{ { 1.0, 0.0, 0.0 }, { 0.0, 1.0 / sqrt( 2.0 ), 1.0 / sqrt( 2.0 ) } };
            state.displacement = { { 0, -0.5, 0 }, { 0, 0.5, 0 } };
            state.momentum     = { gradient_exp.momentum, gradient_exp.momentum };

            hamiltonian.Gradient( state, gradient );
            check_gradient( gradient, gradient_exp );

            hamiltonian.Energy_per_Spin( state, energy_per_spin );
            check_energy( energy_per_spin, eps_expected );
        };
    }
}

TEST_CASE( "Dimer simulation through the (harmonic) lattice Hamiltonian", "[lattice]" )
{
    using namespace Engine::SpinLattice;
    using LatticeHamiltonian = Hamiltonian<
        StateType, Interaction::StandaloneAdaptor<StateType>, Interaction::Lattice_Kinetic,
        Interaction::Lattice_Harmonic_Potential>;

    auto hamiltonian = HamiltonianVariant( LatticeHamiltonian(
        /*geometry=*/make_dimer_geometry(),
        /*bounary_conditions=*/intfield{ 0, 0, 0 },
        typename Interaction::Lattice_Kinetic::Data(
            /*indices=*/{ 0 }, /*magnitudes=*/{ 1.0, 1.0, 1.0 },
            /*normals=*/{ Vector3{ 1, 0, 0 }, Vector3{ 0, 1, 0 }, Vector3{ 0, 0, 1 } } ),
        typename Interaction::Lattice_Harmonic_Potential::Data(
            /*pairs=*/pairfield{ Pair{ 0, 0, { 1, 0, 0 } } },
            /*normals=*/vectorfield{ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } },
            /*magnitudes=*/scalarfield{ 1.0, 1.0, 1.0 } ) ) );

    const auto nos = hamiltonian.get_geometry().nos;

    static constexpr auto labels = make_quantity<std::string_view>( "spin", "displacement", "momentum" );

    SECTION( "Energy and Gradients" )
    {
        const auto check_gradient
            = [nos]( const std::string_view label, const vectorfield & gradient, const vectorfield & gradient_exp )
        {
            for( int i = 0; i < nos; ++i )
            {
                INFO( fmt::format( "{} gradient, i={}", label, i ) );
                INFO( fmt::format( "(found)    = {}", gradient[i].transpose() ) );
                INFO( fmt::format( "(expected) = {}", gradient_exp[i].transpose() ) );
                REQUIRE_THAT( ( gradient[i] - gradient_exp[i] ).norm(), WithinAbs( 0, epsilon_2 ) );
            }
        };

        const auto check_energy = [nos]( const scalarfield & energy_per_spin, const scalar eps_expected )
        {
            for( int i = 0; i < nos; ++i )
            {
                INFO( fmt::format( "Energy per spin, i={}", i ) );
                INFO( fmt::format( "(found)    = {}", energy_per_spin[i] ) );
                INFO( fmt::format( "(expected) = {}", eps_expected ) );
                REQUIRE_THAT( energy_per_spin[i], WithinAbs( eps_expected, epsilon_2 ) );
            }
        };

        auto state           = make_quantity( vectorfield{} );
        auto gradient        = make_quantity( vectorfield( nos, Vector3::Zero() ) );
        auto energy_per_spin = scalarfield( nos, 0.0 );

        SECTION( "test case 1" )
        {
            const quantity<vectorfield> gradient_exp{ vectorfield( 2, Vector3::Zero() ),
                                                      vectorfield{ { 1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 } },
                                                      vectorfield{ { 0.5, 0.0, 0.0 }, { 0.5, 0.0, 0.0 } } };
            constexpr scalar eps_expected = /*spin*/ 0.0 + /*displacement*/ 0.5 + /*momentum*/ 0.5 * ( 0.5 * 0.5 );

            state.spin         = vectorfield( 2, Vector3::Zero() );
            state.displacement = { gradient_exp.displacement[1], gradient_exp.displacement[0] };
            state.momentum     = gradient_exp.momentum;

            hamiltonian.Energy_per_Spin( state, energy_per_spin );
            check_energy( energy_per_spin, eps_expected );

            hamiltonian.Gradient( state, gradient );
            check_gradient( labels.spin, gradient.spin, gradient_exp.spin );
            check_gradient( labels.displacement, gradient.displacement, gradient_exp.displacement );
            check_gradient( labels.momentum, gradient.momentum, gradient_exp.momentum );
        };

        SECTION( "test case 2" )
        {
            const quantity<vectorfield> gradient_exp{ vectorfield( 2, Vector3::Zero() ),
                                                      vectorfield{ { 0, 0.5, 0 }, { 0, 0.5, 0 } },
                                                      vectorfield{ { 0.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0 } } };
            constexpr scalar eps_expected = /*spin*/ 0.0 + /*displacement*/ 0.125 + /*momentum*/ 0.5 * ( 1.0 * 1.0 );

            state.spin         = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } };
            state.displacement = { gradient_exp.displacement[1], gradient_exp.displacement[0] };
            state.momentum     = gradient_exp.momentum;

            hamiltonian.Gradient( state, gradient );
            check_gradient( labels.spin, gradient.spin, gradient_exp.spin );
            check_gradient( labels.displacement, gradient.displacement, gradient_exp.displacement );
            check_gradient( labels.momentum, gradient.momentum, gradient_exp.momentum );

            hamiltonian.Energy_per_Spin( state, energy_per_spin );
            check_energy( energy_per_spin, eps_expected );
        };

        SECTION( "test case 3" )
        {
            const quantity<vectorfield> gradient_exp{ vectorfield( 2, Vector3::Zero() ),
                                                      vectorfield{ { 0, 0.5, 0 }, { 0, -0.5, 0 } },
                                                      vectorfield{ { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 } } };
            const scalar eps_expected = /*spin*/ 0.0 + /*displacement*/ -0.125 + /*momentum*/ 0.5 * ( 1.0 * 1.0 );

            state.spin         = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0 / sqrt( 2.0 ), 1.0 / sqrt( 2.0 ) } };
            state.displacement = { gradient_exp.displacement[1], gradient_exp.displacement[0] };
            state.momentum     = gradient_exp.momentum;

            hamiltonian.Gradient( state, gradient );
            check_gradient( labels.spin, gradient.spin, gradient_exp.spin );
            check_gradient( labels.displacement, gradient.displacement, gradient_exp.displacement );
            check_gradient( labels.momentum, gradient.momentum, gradient_exp.momentum );

            hamiltonian.Energy_per_Spin( state, energy_per_spin );
            check_energy( energy_per_spin, eps_expected );
        };
    }
}
#endif
