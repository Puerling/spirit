#pragma once

#include <engine/spin/StateType.hpp>
#include <engine/spin_lattice/StateType.hpp>

namespace Engine
{

#ifdef SPIRIT_ENABLE_LATTICE
using Engine::SpinLattice::Field;
using Engine::SpinLattice::get;
using Engine::SpinLattice::quantity;
using Engine::SpinLattice::StateType;
#else
using Engine::Spin::Field;
using Engine::Spin::get;
using Engine::Spin::quantity;
using Engine::Spin::StateType;
#endif

} // namespace Engine
