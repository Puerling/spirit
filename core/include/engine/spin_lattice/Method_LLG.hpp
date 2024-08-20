#pragma once

#include <Spirit/Spirit_Defines.h>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Spin_System.hpp>
#include <engine/common/Method_LLG.hpp>
#include <engine/spin_lattice/Hamiltonian.hpp>
#include <engine/spin_lattice/Method_Solver.hpp>

#include <vector>

namespace Engine
{

namespace SpinLattice
{

namespace Common
{

template<Engine::Common::Solver solver>
struct Method_LLG : Engine::Common::Method_LLG<solver>
{
    constexpr Method_LLG( const int nos ) : Engine::Common::Method_LLG<solver>( nos ), xi( nos, Vector3::Zero() ) {};

    void Prepare_Thermal_Field( Data::Parameters_Method_LLG & parameters, const Data::Geometry & geometry )
    {
        namespace Constants = Utility::Constants;

        Engine::Common::Method_LLG<solver>::Prepare_Thermal_Field( parameters, geometry );
        const auto damping = parameters.lattice_damping;

        const scalar epsilon
            = std::sqrt( 2 * parameters.dt * damping * Constants::k_B * parameters.lattice_temperature );

        auto distribution = std::normal_distribution<scalar>{ 0, 1 };

        if( parameters.lattice_temperature > 0 )
        {
            // TODO: parallelization of this is actually not quite so trivial
            // #pragma omp parallel for
            for( std::size_t i = 0; i < xi.size(); ++i )
            {
                for( int dim = 0; dim < 3; ++dim )
                    if( geometry.n_cells[dim] > 1 )
                        xi[i][dim] = epsilon / std::sqrt( geometry.inverse_mass[i] ) * distribution( parameters.prng );
                    else
                        xi[i][dim] = 0;
            }
        }
    }

    // Langevin thermostat
    void Virtual_Force_Momentum(
        const Data::Parameters_Method_LLG & parameters, const Data::Geometry &, const intfield &,
        const vectorfield & momentum, const vectorfield & displacement_force, vectorfield & force_virtual )
    {
        namespace Constants = Utility::Constants;

        scalar dt            = parameters.dt;
        const scalar damping = parameters.lattice_damping;

        if( parameters.direct_minimization || solver == Engine::Common::Solver::VP )
        {
            Vectormath::set_c_a( dt, displacement_force, force_virtual );
        }
        // Dynamics simulation
        else
        {
            Backend::transform(
                SPIRIT_PAR Backend::make_zip_iterator( displacement_force.begin(), momentum.begin(), xi.begin() ),
                Backend::make_zip_iterator( displacement_force.end(), momentum.end(), xi.end() ), force_virtual.begin(),
                Backend::make_zip_function(
                    [dt, damping] SPIRIT_LAMBDA( const Vector3 & f, const Vector3 p, const Vector3 xi ) -> Vector3
                    { return dt * ( f - damping * p ) + xi; } ) );
        }
    }

private:
    vectorfield xi;
};

} // namespace Common

/*
    The Landau-Lifshitz-Gilbert (LLG) method
*/

template<Solver solver>
class Method_LLG : public Method_Solver<solver>
{
public:
    // Constructor
    Method_LLG( std::shared_ptr<system_t> system, int idx_img, int idx_chain );

    double get_simulated_time() override;

    // Method name as string
    std::string_view Name() override;

    // Prepare random numbers for thermal fields, if needed
    void Prepare_Thermal_Field() override;

    // Calculate Forces onto Systems
    void Calculate_Force(
        const std::vector<std::shared_ptr<StateType>> & configurations,
        std::vector<quantity<vectorfield>> & forces ) override;
    void Calculate_Force_Virtual(
        const std::vector<std::shared_ptr<StateType>> & configurations,
        const std::vector<quantity<vectorfield>> & forces,
        std::vector<quantity<vectorfield>> & forces_virtual ) override;

private:
    // Check if the Forces are converged
    bool Converged() override;

    // Save the current Step's Data: spins and energy
    void Save_Current( std::string starttime, int iteration, bool initial = false, bool final = false ) override;
    // A hook into the Method before an Iteration of the Solver
    void Hook_Pre_Iteration() override;
    // A hook into the Method after an Iteration of the Solver
    void Hook_Post_Iteration() override;

    // Sets iteration_allowed to false for the corresponding method
    void Finalize() override;

    void Message_Block_Step( std::vector<std::string> & block ) override;
    void Message_Block_End( std::vector<std::string> & block ) override;

    std::vector<SpinLattice::Common::Method_LLG<common_solver( solver )>> common_methods;

    // Last calculated forces
    std::vector<quantity<vectorfield>> Gradient;
    // Convergence parameters
    std::vector<bool> force_converged;

    // Current energy
    scalar current_energy = 0;

    // Measure of simulated time in picoseconds
    double picoseconds_passed = 0;
};

} // namespace SpinLattice

} // namespace Engine
