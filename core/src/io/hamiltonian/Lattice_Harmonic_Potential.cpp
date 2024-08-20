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
using Engine::SpinLattice::Interaction::Lattice_Harmonic_Potential;

void Lattice_Harmonic_Potential_from_File(
    const std::string & lattice_potential_file, const Data::Geometry & geometry, int & n_indices,
    Lattice_Harmonic_Potential::Data & data )
try
{
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading lattice harmonic potential from file {}", lattice_potential_file ) );

    // parser initialization
    using LatticeHarmonicPotentialTableParser = TableParser<
        int, int, int, int, int, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar,
        scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar>;
    const LatticeHarmonicPotentialTableParser parser( { "i",   "j",   "da",  "db",  "dc",  "k1",  "k2",  "k3",  "k1x",
                                                        "k1y", "k1z", "k1a", "k1b", "k1c", "k2x", "k2y", "k2z", "k2a",
                                                        "k2b", "k2c", "k3x", "k3y", "k3z", "k3a", "k3b", "k3c" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&lattice_potential_file, &geometry]( const std::map<std::string_view, int> & idx )
    {
        std::array K_xyz{ false, false, false };
        std::array K_abc{ false, false, false };
        std::array K_magnitude{ false, false, false };

        if( idx.at( "k1x" ) >= 0 && idx.at( "k1y" ) >= 0 && idx.at( "k1z" ) >= 0 )
            K_xyz[0] = true;
        if( idx.at( "k1a" ) >= 0 && idx.at( "k1b" ) >= 0 && idx.at( "k1c" ) >= 0 )
            K_abc[0] = true;
        if( idx.at( "k1" ) >= 0 )
            K_magnitude[0] = true;

        if( idx.at( "k2x" ) >= 0 && idx.at( "k2y" ) >= 0 && idx.at( "k2z" ) >= 0 )
            K_xyz[1] = true;
        if( idx.at( "k2a" ) >= 0 && idx.at( "k2b" ) >= 0 && idx.at( "k2c" ) >= 0 )
            K_abc[1] = true;
        if( idx.at( "k2" ) >= 0 )
            K_magnitude[1] = true;

        if( idx.at( "k3x" ) >= 0 && idx.at( "k3y" ) >= 0 && idx.at( "k3z" ) >= 0 )
            K_xyz[2] = true;
        if( idx.at( "k3a" ) >= 0 && idx.at( "k3b" ) >= 0 && idx.at( "k3c" ) >= 0 )
            K_abc[2] = true;
        if( idx.at( "k3" ) >= 0 )
            K_magnitude[2] = true;

        const bool valid_vectors = [K_xyz, K_abc]
        {
            for( auto i = 0; i < 3; ++i )
                if( !( K_xyz[i] || K_abc[i] ) )
                    return false;
            return true;
        }();

        const bool valid_pairs = idx.at( "i" ) >= 0 && idx.at( "j" ) >= 0 && idx.at( "da" ) >= 0 && idx.at( "db" ) >= 0
                                 && idx.at( "dc" ) >= 0;

        if( !valid_vectors || !valid_pairs )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "No lattice harmonic potential data could be found in header of file \"{}\"",
                     lattice_potential_file ) );

        struct Row
        {
            Pair pair;
            std::array<scalar, 3> magnitudes{};
            std::array<Vector3, 3> directions{};
        };

        return
            [K_xyz, K_abc, K_magnitude, &geometry]( const LatticeHarmonicPotentialTableParser::read_row_t & row ) -> Row
        {
            struct Data
            {
                int i, j, da, db, dc;
                scalar k1, k2, k3;
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
            w[0] = K_magnitude[0] ? data.k1 : K[0].norm();
            K[0].normalize();

            if( K_abc[1] )
                K[1] = lattice_to_cartesian( geometry, { data.k2a, data.k2b, data.k2c } );
            else if( K_xyz[1] )
                K[1] = { data.k2x, data.k2y, data.k2z };
            w[1] = K_magnitude[1] ? data.k2 : K[1].norm();
            K[1].normalize();

            if( K_abc[2] )
                K[2] = lattice_to_cartesian( geometry, { data.k3a, data.k3b, data.k3c } );
            else if( K_xyz[2] )
                K[2] = { data.k3x, data.k3y, data.k3z };
            w[2] = K_magnitude[2] ? data.k3 : K[2].norm();
            K[2].normalize();

            return Row{ { data.i, data.j, { data.da, data.db, data.dc } }, w, K };
        };
    };

    const std::string anisotropy_size_id = "n_lattice_harmonic_potential";
    const auto rows = parser.parse( lattice_potential_file, anisotropy_size_id, 22ul, transform_factory );
    n_indices       = rows.size();

    data.pairs = pairfield( 0 );
    data.pairs.reserve( n_indices );

    data.magnitudes = scalarfield( 0 );
    data.magnitudes.reserve( 3 * n_indices );

    data.normals = vectorfield( 0 );
    data.normals.reserve( 3 * n_indices );

    // use unordered_set to detect duplicate pairs
    std::unordered_set<Pair, equiv_hash<Pair>, equivalence<Pair>> seen_pairs{};

    for( auto && [pair, magnitudes, normals] : rows )
    {
        if( !seen_pairs.insert( pair ).second )
        {
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "lattice harmonic potential: ignoring duplicate pair: {}", pair ) );
            continue;
        }

        data.pairs.push_back( pair );
        std::move( begin( magnitudes ), end( magnitudes ), std::back_inserter( data.magnitudes ) );
        std::move( begin( normals ), end( normals ), std::back_inserter( data.normals ) );
    }

    data.pairs.shrink_to_fit();
    data.magnitudes.shrink_to_fit();
    data.normals.shrink_to_fit();
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropies from file \"{}\"", lattice_potential_file ) );
}

} // namespace

void Lattice_Harmonic_Potential_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    Lattice_Harmonic_Potential::Data & data )
{

    std::string lattice_harmonic_potential_file{};
    bool lattice_kinetic_from_file = false;
    int n_pairs                    = 0;

    try
    {

        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Anisotropy
        if( config_file_handle.Find( "n_lattice_harmonic_potential" ) )
            lattice_harmonic_potential_file = config_file_name;
        else if( config_file_handle.Find( "lattice_harmonic_potential_file" ) )
            config_file_handle >> lattice_harmonic_potential_file;

        if( !lattice_harmonic_potential_file.empty() )
        {
            // The file name should be valid so we try to read it
            Lattice_Harmonic_Potential_from_File( lattice_harmonic_potential_file, geometry, n_pairs, data );

            lattice_kinetic_from_file = true;
        }

        if( lattice_kinetic_from_file )
            parameter_log.emplace_back( fmt::format(
                "    lattice harmonic potential tensor from file \"{}\"", lattice_harmonic_potential_file ) );
    }
    catch( ... )
    {
        spirit_handle_exception_core( fmt::format(
            "Unable to read lattice harmonic potential tensor from config file \"{}\"", config_file_name ) );
    }
}

} // namespace IO
