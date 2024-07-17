#pragma once

#include <engine/Indexing.hpp>
#include <engine/spin/interaction/Exchange.hpp>
#include <engine/spin_lattice/StateType.hpp>
#include <engine/spin_lattice/interaction/Functor_Prototypes.hpp>

#include <Eigen/Dense>

namespace Engine
{

namespace SpinLattice
{

namespace Interaction
{

// reuse most types from the spin version
struct Exchange : public Spin::Interaction::Exchange
{
    // expected by the SpinWrapper::Functor types to construct the corresponding spin functors
    using base_t = Spin::Interaction::Exchange;

    using state_t = StateType;

    using Energy = Functor::Local::SpinWrapper::Energy_Functor<Exchange>;

    template<Field field>
    using Gradient = Functor::Local::SpinWrapper::Gradient_Functor<field, Exchange>;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    using Energy_Single_Spin = Functor::Local::Energy_Single_Spin_Functor<Energy, 2>;
    // overwrite the Hessian such that it doesn't get used accidentally
    using Hessian = void;

    // Interaction name as string
    static constexpr std::string_view name = "Exchange (spin-lattice)";
};

} // namespace Interaction

} // namespace SpinLattice

} // namespace Engine
