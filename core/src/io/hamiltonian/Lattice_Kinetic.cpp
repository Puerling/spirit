#include <engine/Vectormath_Defines.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>
#include <io/hamiltonian/Interactions.hpp>

#include <unordered_set>
#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

namespace
{

using Data::lattice_to_cartesian;
using Engine::SpinLattice::Interaction::Lattice_Kinetic;

void Lattice_Kinetic_from_File(
    const std::string & lattice_kinetic_file, const Data::Geometry & geometry, int & n_indices,
    Lattice_Kinetic::Data & data )
try
{
    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading mass tensor from file {}", lattice_kinetic_file ) );

    // parser initialization
    using LatticeKineticTableParser = TableParser<
        int, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar,
        scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar>;
    const LatticeKineticTableParser parser( { "i",   "w1",  "w2",  "w3",  "k1x", "k1y", "k1z", "k1a",
                                              "k1b", "k1c", "k2x", "k2y", "k2z", "k2a", "k2b", "k2c",
                                              "k3x", "k3y", "k3z", "k3a", "k3b", "k3c" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&lattice_kinetic_file, &geometry]( const std::map<std::string_view, int> & idx )
    {
        std::array K_xyz{ false, false, false };
        std::array K_abc{ false, false, false };
        std::array K_magnitude{ false, false, false };

        if( idx.at( "k1x" ) >= 0 && idx.at( "k1y" ) >= 0 && idx.at( "k1z" ) >= 0 )
            K_xyz[0] = true;
        if( idx.at( "k1a" ) >= 0 && idx.at( "k1b" ) >= 0 && idx.at( "k1c" ) >= 0 )
            K_abc[0] = true;
        if( idx.at( "w1" ) >= 0 )
            K_magnitude[0] = true;

        if( idx.at( "k2x" ) >= 0 && idx.at( "k2y" ) >= 0 && idx.at( "k2z" ) >= 0 )
            K_xyz[1] = true;
        if( idx.at( "k2a" ) >= 0 && idx.at( "k2b" ) >= 0 && idx.at( "k2c" ) >= 0 )
            K_abc[1] = true;
        if( idx.at( "w2" ) >= 0 )
            K_magnitude[1] = true;

        if( idx.at( "k3x" ) >= 0 && idx.at( "k3y" ) >= 0 && idx.at( "k3z" ) >= 0 )
            K_xyz[2] = true;
        if( idx.at( "k3a" ) >= 0 && idx.at( "k3b" ) >= 0 && idx.at( "k3c" ) >= 0 )
            K_abc[2] = true;
        if( idx.at( "w3" ) >= 0 )
            K_magnitude[2] = true;

        const bool valid_vectors = [K_xyz, K_abc]
        {
            for( auto i = 0; i < 3; ++i )
                if( !( K_xyz[i] || K_abc[i] ) )
                    return false;
            return true;
        }();

        if( !valid_vectors || idx.at( "i" ) < 0 )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No mass data could be found in header of file \"{}\"", lattice_kinetic_file ) );

        struct Row
        {
            int index = -1;
            std::array<scalar, 3> magnitudes{};
            std::array<Vector3, 3> directions{};
        };

        return [K_xyz, K_abc, K_magnitude, &geometry]( const LatticeKineticTableParser::read_row_t & row ) -> Row
        {
            struct Data
            {
                int i;
                scalar w1, w2, w3;
                scalar k1x, k1y, k1z, k1a, k1b, k1c;
                scalar k2x, k2y, k2z, k2a, k2b, k2c;
                scalar k3x, k3y, k3z, k3a, k3b, k3c;
            };
            const Data data = IO::make_from_tuple<Data>( row );

            std::array<scalar, 3> w{ 1.0, 1.0, 1.0 };
            std::array<Vector3, 3> K{ Vector3{ 1, 0, 0 }, Vector3{ 0, 1, 0 }, Vector3{ 0, 0, 1 } };

            // transform to cartesian and find magnitude
            if( K_abc[0] )
                K[0] = lattice_to_cartesian( geometry, { data.k1a, data.k1b, data.k1c } );
            else if( K_xyz[0] )
                K[0] = { data.k1x, data.k1y, data.k1z };
            w[0] = K_magnitude[0] ? data.w1 : K[0].norm();
            K[0].normalize();

            if( K_abc[1] )
                K[1] = lattice_to_cartesian( geometry, { data.k2a, data.k2b, data.k2c } );
            else if( K_xyz[1] )
                K[1] = { data.k2x, data.k2y, data.k2z };
            w[1] = K_magnitude[1] ? data.w2 : K[1].norm();
            K[1].normalize();

            if( K_abc[2] )
                K[2] = lattice_to_cartesian( geometry, { data.k3a, data.k3b, data.k3c } );
            else if( K_xyz[2] )
                K[2] = { data.k3x, data.k3y, data.k3z };
            w[2] = K_magnitude[2] ? data.w3 : K[2].norm();
            K[2].normalize();

            return Row{ data.i, w, K };
        };
    };

    static constexpr scalar thresh         = 1e-6;
    const std::string inverse_mass_size_id = "n_inverse_mass";
    const auto rows = parser.parse( lattice_kinetic_file, inverse_mass_size_id, 22ul, transform_factory );
    n_indices       = rows.size();

    data.indices = intfield( 0 );
    data.indices.reserve( n_indices );

    data.magnitudes = scalarfield( 0 );
    data.magnitudes.reserve( 3 * n_indices );

    data.normals = vectorfield( 0 );
    data.normals.reserve( 3 * n_indices );

    // use unordered_set to deduplicate indices
    std::unordered_set<int> seen_indices{};

    for( auto && [i, w, K] : rows )
    {
        if( abs( K[0].dot( K[1].cross( K[2] ) ) ) < thresh )
        {
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "discarding degenerate inverse mass tensor at index i={}", i ) );
            continue;
        }

        if( !seen_indices.insert( i ).second )
        {
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "ignoring duplicate for inverse mass tensor at index i={}", i ) );
            continue;
        }

        data.indices.push_back( i );
        std::move( begin( w ), end( w ), std::back_inserter( data.magnitudes ) );
        std::move( begin( K ), end( K ), std::back_inserter( data.normals ) );
    }

    data.indices.shrink_to_fit();
    data.magnitudes.shrink_to_fit();
    data.normals.shrink_to_fit();
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropies from file \"{}\"", lattice_kinetic_file ) );
}

} // namespace

void Lattice_Kinetic_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    Lattice_Kinetic::Data & data )
{

    std::string lattice_kinetic_file{};
    bool lattice_kinetic_from_file = false;
    int n_pairs                    = 0;

    try
    {

        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Anisotropy
        if( config_file_handle.Find( "n_inverse_mass" ) )
            lattice_kinetic_file = config_file_name;
        else if( config_file_handle.Find( "inverse_mass_file" ) )
            config_file_handle >> lattice_kinetic_file;

        if( !lattice_kinetic_file.empty() )
        {
            // The file name should be valid so we try to read it
            Lattice_Kinetic_from_File( lattice_kinetic_file, geometry, n_pairs, data );

            lattice_kinetic_from_file = true;
        }
        else
        {
            const std::size_t n_cell_atoms = geometry.n_cell_atoms;
            const std::array<Vector3, 3> local_basis{ Vector3{ 1, 0, 0 }, Vector3{ 0, 1, 0 }, Vector3{ 0, 0, 1 } };

            if( config_file_handle.Find( "lattice_mass" ) )
            {

                data.indices.reserve( n_cell_atoms );
                data.normals.reserve( 3 * n_cell_atoms );
                data.magnitudes.reserve( 3 * n_cell_atoms );

                for( std::size_t iatom = 0; iatom < n_cell_atoms; ++iatom )
                {
                    scalar mass = 0;
                    if( !( config_file_handle >> mass ) )
                    {
                        Log( Log_Level::Warning, Log_Sender::IO,
                             fmt::format(
                                 "Not enough values specified after 'lattice_mass'. Expected {}. Using mass[{}]=1",
                                 n_cell_atoms, iatom ) );
                        mass = 1.0;
                    }

                    data.indices.push_back( iatom );
                    std::copy( local_basis.begin(), local_basis.end(), std::back_inserter( data.normals ) );
                    for( std::size_t dim = 0; dim < 3; ++dim )
                    {
                        data.magnitudes.push_back( mass );
                    }
                }
            }
            if( !Lattice_Kinetic::valid_data( data ) )
            {
                spirit_throw(
                    Utility::Exception_Classifier::Input_parse_failed, Log_Level::Error,
                    fmt::format( "Invalid mass tensor constructed from config file: \"{}\"", config_file_name ) );
            }
        }

        if( lattice_kinetic_from_file )
            parameter_log.emplace_back(
                fmt::format( "    inverse mass tensor from file \"{}\"", lattice_kinetic_file ) );
        else if( !data.magnitudes.empty() )
        {
            for( int iatom = 0; iatom < geometry.n_cell_atoms; ++iatom )
                parameter_log.emplace_back( fmt::format( "    mass[{}]={}", iatom, data.magnitudes[3 * iatom] ) );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read mass tensor from config file \"{}\"", config_file_name ) );
    }
}

} // namespace IO
