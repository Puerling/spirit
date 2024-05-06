#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_SOLVER_LBFGS_OSO_HPP
#define SPIRIT_CORE_ENGINE_SPIN_SOLVER_LBFGS_OSO_HPP

#include <engine/spin/Method_Solver.hpp>
#include <utility/Constants.hpp>

namespace Engine
{

namespace Spin
{

template<>
class SolverData<Solver::LBFGS_OSO> : public Method
{
protected:
    using Method::Method;
    // General
    static constexpr int n_lbfgs_memory = 3; // how many previous iterations are stored in the memory
    static constexpr scalar maxmove     = Utility::Constants::Pi / 200.0;
    int local_iter;
    scalarfield rho;
    scalarfield alpha;

    // OSO
    std::vector<std::vector<vectorfield>> delta_a;
    std::vector<std::vector<vectorfield>> delta_grad;
    std::vector<vectorfield> searchdir;
    std::vector<vectorfield> grad;
    std::vector<vectorfield> grad_pr;
    std::vector<vectorfield> q_vec;

    // Actual Forces on the configurations
    std::vector<vectorfield> forces_predictor;
    // Virtual Forces used in the Steps
    std::vector<vectorfield> forces_virtual_predictor;

    std::vector<std::shared_ptr<vectorfield>> configurations_predictor;
    std::vector<std::shared_ptr<vectorfield>> configurations_temp;
};

template<>
inline void Method_Solver<Solver::LBFGS_OSO>::Initialize()
{
    using namespace Utility;

    this->delta_a = std::vector<std::vector<vectorfield>>(
        this->noi, std::vector<vectorfield>( n_lbfgs_memory, vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->delta_grad = std::vector<std::vector<vectorfield>>(
        this->noi, std::vector<vectorfield>( n_lbfgs_memory, vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->rho            = scalarfield( n_lbfgs_memory, 0 );
    this->alpha          = scalarfield( n_lbfgs_memory, 0 );
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->searchdir      = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad           = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad_pr        = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->q_vec          = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->local_iter     = 0;
};

/*
    Implemented according to Aleksei Ivanov's paper: https://arxiv.org/abs/1904.02669
    TODO: reference painless conjugate gradients
    See also Jorge Nocedal and Stephen J. Wright 'Numerical Optimization' Second Edition, 2006 (p. 121).
*/

template<>
inline void Method_Solver<Solver::LBFGS_OSO>::Iteration()
{
    // update forces which are -dE/ds
    this->Calculate_Force( this->configurations, this->forces );
    // calculate gradients for OSO
    for( int img = 0; img < this->noi; img++ )
    {
        auto & image    = *this->configurations[img];
        auto & grad_ref = this->grad[img];

        const auto * f = raw_pointer_cast( this->forces[img].data() );
        const auto * s = raw_pointer_cast( image.data() );
        auto * fv      = raw_pointer_cast( this->forces_virtual[img].data() );

        Backend::for_each_n(
            SPIRIT_PAR Backend::make_counting_iterator( 0 ), this->nos,
            [f, s, fv] SPIRIT_LAMBDA( const int idx ) { fv[idx] = s[idx].cross( f[idx] ); } );

        Solver_Kernels::oso_calc_gradients( grad_ref, image, this->forces[img] );
    }

    // calculate search direction
    Solver_Kernels::lbfgs_get_searchdir(
        this->local_iter, this->rho, this->alpha, this->q_vec, this->searchdir, this->delta_a, this->delta_grad,
        this->grad, this->grad_pr, n_lbfgs_memory, maxmove );

    // Scale direction
    scalar scaling = 1;
    for( int img = 0; img < noi; img++ )
        scaling = std::min( Solver_Kernels::maximum_rotation( searchdir[img], maxmove ), scaling );

    for( int img = 0; img < noi; img++ )
    {
        Vectormath::scale( searchdir[img], scaling );
    }

    // rotate spins
    Solver_Kernels::oso_rotate( this->configurations, this->searchdir );
}

template<>
inline std::string Method_Solver<Solver::LBFGS_OSO>::SolverName()
{
    return "LBFGS_OSO";
}

template<>
inline std::string Method_Solver<Solver::LBFGS_OSO>::SolverFullName()
{
    return "Limited memory Broyden-Fletcher-Goldfarb-Shanno using exponential transforms";
}

} // namespace Spin

} // namespace Engine

#endif
