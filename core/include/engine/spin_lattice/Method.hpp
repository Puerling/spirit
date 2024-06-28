#pragma once

#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Method.hpp>
#include <engine/spin_lattice/Hamiltonian.hpp>

namespace Engine
{

namespace SpinLattice
{

using hamiltonian_t = Engine::SpinLattice::HamiltonianVariant;
using system_t      = Data::Spin_System<hamiltonian_t>;
using chain_t       = Data::Spin_System_Chain<hamiltonian_t>;

} // namespace SpinLattice

} // namespace Engine
