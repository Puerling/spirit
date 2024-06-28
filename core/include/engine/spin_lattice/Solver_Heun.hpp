#pragma once
#include <engine/spin_lattice/Method_Solver.hpp>

namespace Engine
{

namespace SpinLattice
{

template<>
class SolverData<Solver::Heun> : public SolverMethods
{
protected:
    using SolverMethods::Calculate_Force;
    using SolverMethods::Calculate_Force_Virtual;
    using SolverMethods::Prepare_Thermal_Field;
    using SolverMethods::SolverMethods;
    // Actual Forces on the configurations
    std::vector<quantity<vectorfield>> forces_predictor;
    // Virtual Forces used in the Steps
    std::vector<quantity<vectorfield>> forces_virtual_predictor;

    std::vector<std::shared_ptr<StateType>> configurations_predictor;
    std::vector<std::shared_ptr<StateType>> delta_configurations;

    template<Field... field>
    void Iteration_Impl( enum_sequence<Field, field...> );
};

template<>
inline void Method_Solver<Solver::Heun>::Initialize()
{
    this->forces         = std::vector( this->noi, make_quantity( vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->forces_virtual = std::vector( this->noi, make_quantity( vectorfield( this->nos, { 0, 0, 0 } ) ) );

    this->forces_predictor         = std::vector( this->noi, make_quantity( vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->forces_virtual_predictor = std::vector( this->noi, make_quantity( vectorfield( this->nos, { 0, 0, 0 } ) ) );

    this->delta_configurations = std::vector<std::shared_ptr<StateType>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        delta_configurations[i] = std::make_shared<StateType>( make_state<StateType>( this->nos ) );

    this->configurations_predictor = std::vector<std::shared_ptr<StateType>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_predictor[i] = std::make_shared<StateType>( make_state<StateType>( this->nos ) );
}

/*
    Template instantiation of the Simulation class for use with the Heun Solver.
        The Heun method is a basic solver for the PDE at hand here. It is sufficiently
        efficient and stable.
    This method is described for spin systems including thermal noise in
        U. Nowak, Thermally Activated Reversal in Magnetic Nanostructures,
        Annual Reviews of Computational Physics IX Chapter III (p 105) (2001)
*/
template<>
inline void Method_Solver<Solver::Heun>::Iteration()
{
    SolverData::Iteration_Impl( make_enum_sequence<Field>() );
}

template<Field... field>
inline void SolverData<Solver::Heun>::Iteration_Impl( enum_sequence<Field, field...> )
{
    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // Get the actual forces on the configurations
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        // First step - Predictor
        ( ..., Solver_Kernels::heun_predictor<field == Field::Spin>(
                   get<field>( *this->configurations[i] ), get<field>( this->forces_virtual[i] ),
                   get<field>( *this->delta_configurations[i] ), get<field>( *this->configurations_predictor[i] ) ) );
    }

    // Calculate_Force for the Corrector
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Corrector step for each image
    for( int i = 0; i < this->noi; i++ )
    {
        // Second step - Corrector
        ( ..., Solver_Kernels::heun_corrector<field == Field::Spin>(
                   get<field>( this->forces_virtual_predictor[i] ), get<field>( *this->delta_configurations[i] ),
                   get<field>( *this->configurations_predictor[i] ), get<field>( *this->configurations[i] ) ) );
    }
}

} // namespace SpinLattice

} // namespace Engine
