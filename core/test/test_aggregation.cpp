#include <Spirit/Configurations.h>
#include <Spirit/Constants.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Parameters_LLG.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>
#include <data/State.hpp>
#include <engine/StateType.hpp>
#include <engine/Vectormath.hpp>

#include "catch.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iomanip>
#include <iostream>
#include <sstream>

using Catch::Matchers::WithinAbs;

using Engine::Field;
using Engine::get;
using Engine::StateType;

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

TEST_CASE( "Ensure that Hamiltonian is really just an aggregator", "[aggregation]" )
{
    // Hamiltonians to be tested
    std::vector<const char *> hamiltonian_input_files{
#ifndef SPIRIT_ENABLE_LATTICE
        "core/test/input/fd_gaussian.cfg",
        "core/test/input/fd_pairs.cfg",
        "core/test/input/fd_neighbours.cfg",
#else
        "core/test/input/lattice_spring.cfg"
#endif
    };

    for( const auto * input_file : hamiltonian_input_files )
    {
        INFO( " Testing" << input_file );

        auto simulation_state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
        Configuration_Random( simulation_state.get() );
        const auto & state = *simulation_state->active_image->state;
        auto & hamiltonian = simulation_state->active_image->hamiltonian;
        auto nos           = get<Field::Spin>( state ).size();

        if( hamiltonian->active_count() == 0 )
        {
            CAPTURE( fmt::format( " Warning: input file \"{}\" didn't specify any interaction to test.", input_file ) );
        }

        auto active_interactions = hamiltonian->active_interactions();
        auto aggregator          = [&active_interactions]( const auto init, const auto & f )
        { return std::accumulate( std::begin( active_interactions ), std::end( active_interactions ), init, f ); };

        scalar energy_hamiltonian = hamiltonian->Energy( state );
        scalar energy_aggregated  = aggregator(
            0.0, [&state]( const scalar v, const auto & interaction ) -> scalar
            { return v + interaction->Energy( state ); } );

        INFO( "Hamiltonian::Energy" )
        INFO( "[total], epsilon = " << epsilon_2 << "\n" );
        INFO( "Energy (Hamiltonian) = " << energy_hamiltonian << "\n" );
        INFO( "Energy (aggregated)  = " << energy_aggregated << "\n" );
        REQUIRE_THAT( energy_hamiltonian, WithinAbs( energy_aggregated, epsilon_2 ) );

        scalarfield energy_per_spin_hamiltonian{}; // resize and clear should be handled by the hamiltonian
        hamiltonian->Energy_per_Spin( state, energy_per_spin_hamiltonian );
        scalarfield energy_per_spin_aggregated = aggregator(
            scalarfield( nos, 0 ),
            [&state]( const scalarfield & v, const auto & interaction ) -> scalarfield
            {
                const auto nos       = get<Field::Spin>( state ).size();
                auto energy_per_spin = scalarfield( nos, 0 );
                interaction->Energy_per_Spin( state, energy_per_spin );
#pragma omp parallel for
                for( std::size_t i = 0; i < nos; ++i )
                    energy_per_spin[i] += v[i];

                return energy_per_spin;
            } );

        for( int i = 0; i < simulation_state->nos; i++ )
        {
            INFO( "Hamiltonian::Energy_per_Spin" )
            INFO( "i = " << i << ", epsilon = " << epsilon_2 << "\n" );
            INFO( "Energy (Hamiltonian)   = " << energy_per_spin_hamiltonian[i] << "\n" );
            INFO( "Energy (aggregated) = " << energy_per_spin_aggregated[i] << "\n" );
            REQUIRE_THAT( energy_per_spin_hamiltonian[i], WithinAbs( energy_per_spin_aggregated[i], epsilon_2 ) );
        }

#ifndef SPIRIT_ENABLE_LATTICE
        auto gradient_hamiltonian = vectorfield( 0 ); // resize and clear should be handled by the hamiltonian
        hamiltonian->Gradient( state, gradient_hamiltonian );
        vectorfield gradient_aggregated = aggregator(
            vectorfield( nos, Vector3::Zero() ),
            [&state]( const vectorfield & v, const auto & interaction ) -> vectorfield
            {
                auto gradient = vectorfield( get<Field::Spin>( state ).size(), Vector3::Zero() );
                interaction->Gradient( state, gradient );
                Engine::Vectormath::add_c_a( 1.0, v, gradient );
                return gradient;
            } );

        for( int i = 0; i < simulation_state->nos; ++i )
        {
            INFO( "Hamiltonian::Gradient" )
            INFO( "i = " << i << ", epsilon = " << epsilon_2 << "\n" );
            INFO( "Gradient (Hamiltonian)   = " << gradient_hamiltonian[i] << "\n" );
            INFO( "Gradient (aggregated) = " << gradient_aggregated[i] << "\n" );
            REQUIRE( gradient_hamiltonian[i].isApprox( gradient_aggregated[i], epsilon_2 ) );
        }
#else
        auto gradient_hamiltonian
            = Engine::make_state<Engine::StateType>( 0 ); // resize and clear should be handled by the hamiltonian
        hamiltonian->Gradient( state, gradient_hamiltonian );
        const Engine::SpinLattice::quantity<vectorfield> gradient_aggregated = aggregator(
            Engine::SpinLattice::make_quantity( vectorfield( nos, Vector3::Zero() ) ),
            [&state]( const Engine::SpinLattice::quantity<vectorfield> & v, const auto & interaction )
                -> Engine::SpinLattice::quantity<vectorfield>
            {
                auto gradient = Engine::SpinLattice::make_quantity(
                    vectorfield( state.spin.size(), Vector3::Zero() ),
                    vectorfield( state.displacement.size(), Vector3::Zero() ),
                    vectorfield( state.momentum.size(), Vector3::Zero() ) );
                interaction->Gradient( state, gradient );
                Engine::Vectormath::add_c_a( 1.0, v.spin, gradient.spin );
                Engine::Vectormath::add_c_a( 1.0, v.displacement, gradient.displacement );
                Engine::Vectormath::add_c_a( 1.0, v.momentum, gradient.momentum );
                return gradient;
            } );

        {
            const auto test = [nos](
                                  const std::string_view label, const vectorfield & grad_hamiltonian,
                                  const vectorfield & grad_aggregated )
            {
                for( unsigned int i = 0; i < nos; i++ )
                {
                    INFO( label << ( label.empty() ? "" : "\n" ) );
                    INFO( "i = " << i << ", epsilon = " << epsilon_2 << "\n" );
                    INFO( "Gradient (Hamiltonian) = " << grad_hamiltonian[i].transpose() << "\n" );
                    INFO( "Gradient (aggregated)  = " << grad_aggregated[i].transpose() << "\n" );
                    REQUIRE( grad_hamiltonian[i].isApprox( grad_aggregated[i], epsilon_2 ) );
                }
            };

            test( "Hamiltonian::Gradient<Field::Spin>", gradient_hamiltonian.spin, gradient_aggregated.spin );
            test(
                "Hamiltonian::Gradient<Field::Displacement>", gradient_hamiltonian.displacement,
                gradient_aggregated.displacement );
            test(
                "Hamiltonian::Gradient<Field::Momentum>", gradient_hamiltonian.momentum, gradient_aggregated.momentum );
        }
#endif

        scalar energy_combined_hamiltonian = 0;
        auto gradient_combined_hamiltonian = Engine::make_state<Engine::StateType>( 0 );
        hamiltonian->Gradient_and_Energy( state, gradient_combined_hamiltonian, energy_combined_hamiltonian );

#ifndef SPIRIT_ENABLE_LATTICE
        for( int i = 0; i < simulation_state->nos; ++i )
        {
            INFO( "Hamiltonian::Gradient_and_Energy" )
            INFO( "i = " << i << ", epsilon = " << epsilon_2 << "\n" );
            INFO( "Gradient (combined)   = " << gradient_combined_hamiltonian[i] << "\n" );
            INFO( "Gradient (aggregated) = " << gradient_aggregated[i] << "\n" );
            REQUIRE( gradient_combined_hamiltonian[i].isApprox( gradient_aggregated[i], epsilon_2 ) );
        }
#else
        {
            const auto test = [nos](
                                  const std::string_view label, const vectorfield & grad_hamiltonian,
                                  const vectorfield & grad_aggregated )
            {
                for( unsigned int i = 0; i < nos; i++ )
                {
                    INFO( label << ( label.empty() ? "" : "\n" ) );
                    INFO( "i = " << i << ", epsilon = " << epsilon_2 << "\n" );
                    INFO( "Gradient (combined)    = " << grad_hamiltonian[i].transpose() << "\n" );
                    INFO( "Gradient (aggregated)  = " << grad_aggregated[i].transpose() << "\n" );
                    REQUIRE( grad_hamiltonian[i].isApprox( grad_aggregated[i], epsilon_2 ) );
                }
            };
            test(
                "Hamiltonian::Gradient_and_Energy<Field::Spin>", gradient_combined_hamiltonian.spin,
                gradient_aggregated.spin );
            test(
                "Hamiltonian::Gradient_and_Energy<Field::Displacement>", gradient_combined_hamiltonian.displacement,
                gradient_aggregated.displacement );
            test(
                "Hamiltonian::Gradient_and_Energy<Field::Momentum>", gradient_combined_hamiltonian.momentum,
                gradient_aggregated.momentum );
        }

#endif
        INFO( "Hamiltonian::Gradient_and_Energy" )
        INFO( "[total], epsilon = " << epsilon_2 << "\n" );
        INFO( "Energy (combined)   = " << energy_combined_hamiltonian << "\n" );
        INFO( "Energy (aggregated) = " << energy_aggregated << "\n" );
        REQUIRE_THAT( energy_combined_hamiltonian, WithinAbs( energy_aggregated, epsilon_2 ) );
    }
}
