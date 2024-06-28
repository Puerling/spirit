#pragma once
#include <engine/spin_lattice/Method_Solver.hpp>

namespace Engine
{

namespace SpinLattice
{

template<>
class SolverData<Solver::RungeKutta4> : public SolverMethods
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

    std::vector<std::shared_ptr<StateType>> configurations_k1;
    std::vector<std::shared_ptr<StateType>> configurations_k2;
    std::vector<std::shared_ptr<StateType>> configurations_k3;

    template<Field... field>
    void Iteration_Impl( enum_sequence<Field, field...> );
};

template<>
inline void Method_Solver<Solver::RungeKutta4>::Initialize()
{
    this->forces         = std::vector( this->noi, make_quantity( vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->forces_virtual = std::vector( this->noi, make_quantity( vectorfield( this->nos, { 0, 0, 0 } ) ) );

    this->forces_predictor         = std::vector( this->noi, make_quantity( vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->forces_virtual_predictor = std::vector( this->noi, make_quantity( vectorfield( this->nos, { 0, 0, 0 } ) ) );

    this->configurations_predictor = std::vector<std::shared_ptr<StateType>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_predictor[i] = std::make_shared<StateType>( make_state<StateType>( this->nos ) );

    this->configurations_k1 = std::vector<std::shared_ptr<StateType>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_k1[i] = std::make_shared<StateType>( make_state<StateType>( this->nos ) );

    this->configurations_k2 = std::vector<std::shared_ptr<StateType>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_k2[i] = std::make_shared<StateType>( make_state<StateType>( this->nos ) );

    this->configurations_k3 = std::vector<std::shared_ptr<StateType>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_k3[i] = std::make_shared<StateType>( make_state<StateType>( this->nos ) );
}

template<>
inline void Method_Solver<Solver::RungeKutta4>::Iteration()
{
    SolverData<Solver::RungeKutta4>::Iteration_Impl( make_enum_sequence<Field>() );
}
/*
    Template instantiation of the Simulation class for use with the 4th order Runge Kutta Solver.
*/
template<Field... field>
inline void SolverData<Solver::RungeKutta4>::Iteration_Impl( enum_sequence<Field, field...> )
{
    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // Get the actual forces on the configurations
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        ( ..., Solver_Kernels::rk4_predictor_1<field == Field::Spin>(
                   get<field>( *this->configurations[i] ), get<field>( this->forces_virtual[i] ),
                   get<field>( *this->configurations_k1[i] ), get<field>( *this->configurations_predictor[i] ) ) );
    }

    // Calculate_Force for the predictor
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        ( ..., Solver_Kernels::rk4_predictor_2<field == Field::Spin>(
                   get<field>( *this->configurations[i] ), get<field>( this->forces_virtual_predictor[i] ),
                   get<field>( *this->configurations_k2[i] ), get<field>( *this->configurations_predictor[i] ) ) );
    }

    // Calculate_Force for the predictor (k3)
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        ( ..., Solver_Kernels::rk4_predictor_3<field == Field::Spin>(
                   get<field>( *this->configurations[i] ), get<field>( this->forces_virtual_predictor[i] ),
                   get<field>( *this->configurations_k3[i] ), get<field>( *this->configurations_predictor[i] ) ) );
    }

    // Calculate_Force for the predictor (k4)
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Corrector step for each image
    for( int i = 0; i < this->noi; i++ )
    {
        ( ..., Solver_Kernels::rk4_corrector<field == Field::Spin>(
                   get<field>( forces_virtual_predictor[i] ), get<field>( *configurations_k1[i] ),
                   get<field>( *configurations_k2[i] ), get<field>( *configurations_k3[i] ),
                   get<field>( *configurations_predictor[i] ), get<field>( *configurations[i] ) ) );
    }
}

} // namespace SpinLattice

} // namespace Engine
