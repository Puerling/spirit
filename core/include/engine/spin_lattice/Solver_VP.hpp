#pragma once
#include <engine/spin_lattice/Method_Solver.hpp>

namespace Engine
{

namespace SpinLattice
{

template<>
class SolverData<Solver::VP> : public SolverMethods
{
protected:
    using SolverMethods::Calculate_Force;
    using SolverMethods::Calculate_Force_Virtual;
    using SolverMethods::Prepare_Thermal_Field;
    using SolverMethods::SolverMethods;
    // Actual Forces on the configurations
    std::vector<quantity<vectorfield>> forces;
    // Virtual Forces used in the Steps
    std::vector<quantity<vectorfield>> forces_virtual;

    std::vector<std::shared_ptr<StateType>> configurations;

    // "Mass of our particle" which we accelerate
    static constexpr scalar mass = 1.0;

    // Force in previous step [noi][nos]
    std::vector<quantity<vectorfield>> forces_previous;
    // Velocity in previous step [noi][nos]
    std::vector<quantity<vectorfield>> velocities_previous;
    // Velocity used in the Steps [noi][nos]
    std::vector<quantity<vectorfield>> velocities;
    // Projection of velocities onto the forces [noi]
    std::vector<quantity<scalar>> projection;
    // |force|^2
    std::vector<quantity<scalar>> force_norm2;

    std::vector<std::shared_ptr<const Data::Parameters_Method_LLG>> llg_parameters;

    template<Field... field>
    void Iteration_Impl( enum_sequence<Field, field...> );
};

template<>
inline void Method_Solver<Solver::VP>::Initialize()
{
    this->forces         = std::vector( this->noi, make_quantity( vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->forces_virtual = std::vector( this->noi, make_quantity( vectorfield( this->nos, { 0, 0, 0 } ) ) );

    this->velocities
        = std::vector( this->noi, make_quantity( vectorfield( this->nos, Vector3::Zero() ) ) ); // [noi][nos]
    this->velocities_previous = velocities;                                                     // [noi][nos]
    this->forces_previous     = velocities;                                                     // [noi][nos]
    this->projection          = std::vector( this->noi, make_quantity<scalar>( 0 ) );           // [noi]
    this->force_norm2         = std::vector( this->noi, make_quantity<scalar>( 0 ) );           // [noi]

    this->llg_parameters = std::vector<std::shared_ptr<const Data::Parameters_Method_LLG>>( this->noi, nullptr );
    for( int i = 0; i < this->noi; i++ )
        this->llg_parameters[i] = this->systems[i]->llg_parameters;
}

template<>
inline void Method_Solver<Solver::VP>::Iteration()
{
    SolverData<Solver::VP>::Iteration_Impl( make_enum_sequence<Field>() );
}

/*
    Template instantiation of the Simulation class for use with the VP Solver.
        The velocity projection method is often efficient for direct minimization,
        but deals poorly with quickly varying fields or stochastic noise.
    Paper: P. F. Bessarab et al., Method for finding mechanism and activation energy
           of magnetic transitions, applied to skyrmion and antivortex annihilation,
           Comp. Phys. Comm. 196, 335 (2015).
*/
template<Field... field>
inline void SolverData<Solver::VP>::Iteration_Impl( enum_sequence<Field, field...> )
{
    // Set previous
    for( int i = 0; i < noi; ++i )
    {
        ( ..., Backend::copy(
                   SPIRIT_PAR get<field>( forces[i] ).begin(), get<field>( forces[i] ).end(),
                   get<field>( forces_previous[i] ).begin() ) );
    }

    // Get the forces on the configurations
    this->Calculate_Force( configurations, forces );
    this->Calculate_Force_Virtual( configurations, forces, forces_virtual );

    for( int i = 0; i < noi; ++i )
    {
        ( ..., Solver_Kernels::VP::bare_velocity(
                   get<field>( forces[i] ), get<field>( forces_previous[i] ), get<field>( velocities[i] ) ) );
    }

    // summation over the full state space
    const Vector2 projections = Backend::cpu::transform_reduce(
        velocities.begin(), velocities.end(), forces.begin(), Vector2{ 0.0, 0.0 }, Backend::plus<Vector2>{},
        []( const quantity<vectorfield> & velocity, const quantity<vectorfield> & force ) -> Vector2
        {
            return (
                Vector2::Zero() + ...
                + Vector2{ Vectormath::dot( get<field>( velocity ), get<field>( force ) ),
                           Vectormath::dot( get<field>( force ), get<field>( force ) ) } );
        } );

    for( int i = 0; i < noi; ++i )
    {
        ( ...,
          [this, i, &projections]
          {
              // Calculate the projected velocity
              Solver_Kernels::VP::projected_velocity(
                  projections, get<field>( forces[i] ), get<field>( velocities[i] ) );

              // Apply the projected velocity
              Solver_Kernels::VP::apply_velocity<field == Field::Spin>(
                  get<field>( velocities[i] ), get<field>( forces[i] ), llg_parameters[i]->dt,
                  get<field>( *configurations[i] ) );
          }() );
    }
}

} // namespace SpinLattice

} // namespace Engine
