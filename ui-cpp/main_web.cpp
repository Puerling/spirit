#ifdef __EMSCRIPTEN__

#include <main_window.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/IO.h>
#include <Spirit/Log.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/Transitions.h>
#include <Spirit/Version.h>

#include <fmt/format.h>

#include <iostream>
#include <memory>
#include <string>

// Initialise global state pointer
std::shared_ptr<State> state;

// Main
int main( int argc, char ** argv )
try
{
    std::cout << "--------------------------------------\n";
    std::cout << "Spirit Version: " << Spirit_Version_Full() << "\n";
    std::cout << Spirit_Compiler_Full() << "\n";
    std::cout << "--------------------------------------\n";
    std::cout << "scalar_type = " << Spirit_Scalar_Type() << "\n";
    std::cout << "Parallelisation:\n";
    std::cout << "   - OpenMP  = " << Spirit_OpenMP() << "\n";
    std::cout << "   - Cuda    = " << Spirit_Cuda() << "\n";
    std::cout << "   - Threads = " << Spirit_Threads() << "\n";
    std::cout << "Other:\n";
    std::cout << "   - Defects = " << Spirit_Defects() << "\n";
    std::cout << "   - Pinning = " << Spirit_Pinning() << "\n";
    std::cout << "   - FFTW    = " << Spirit_FFTW() << "\n";

    // Default options
    bool quiet          = false;
    std::string cfgfile = "";

    // Initialise State
    state = std::shared_ptr<State>( State_Setup( cfgfile.c_str(), quiet ), State_Delete );
    Log_Set_Output_File_Tag( state.get(), "" );

    int n_cells[3] = { 30, 30, 10 };
    Geometry_Set_N_Cells( state.get(), n_cells );

    // Standard Initial spin configuration
    Configuration_Random( state.get() );

    float normal[3] = { 0, 0, 1 };
    Hamiltonian_Set_Field( state.get(), 5, normal );
    float jij[1] = { 1 };
    Hamiltonian_Set_Exchange( state.get(), 1, jij );
    float dij[1] = { 0.3f };
    Hamiltonian_Set_DMI( state.get(), 1, dij );

    ui::MainWindow window( state );

    // Open the Application
    int exec = window.run();
    // If Application is closed normally
    if( exec != 0 )
        throw exec;
    // Finish
    state.reset();
    return exec;
}
catch( const std::exception & e )
{
    fmt::print( "caught std::exception: {}\n", e.what() );
}

#endif