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

using Engine::SpinLattice::Interaction::Lattice_Spring_Potential;

void Lattice_Spring_Potential_from_File(
    const std::string & lattice_potential_file, const Data::Geometry &, int & n_pairs,
    Lattice_Spring_Potential::Data & data )
try
{
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading lattice spring potential from file {}", lattice_potential_file ) );

    // parser initialization
    using LatticePotentialTableParser = TableParser<int, int, int, int, int, scalar>;
    const LatticePotentialTableParser parser( { "i", "j", "da", "db", "dc", "k" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&lattice_potential_file]( const std::map<std::string_view, int> & idx )
    {
        const bool valid_pairs = idx.at( "i" ) >= 0 && idx.at( "j" ) >= 0 && idx.at( "da" ) >= 0 && idx.at( "db" ) >= 0
                                 && idx.at( "dc" ) >= 0;

        if( !valid_pairs || idx.at( "k" ) < 0 )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "No lattice spring potential data could be found in header of file \"{}\"",
                     lattice_potential_file ) );

        struct Row
        {
            Pair pair{};
            scalar magnitude;
        };

        return []( const LatticePotentialTableParser::read_row_t & row ) -> Row
        {
            struct Data
            {
                int i, j, da, db, dc;
                scalar magnitude;
            };
            const Data data = IO::make_from_tuple<Data>( row );

            return Row{ { data.i, data.j, { data.da, data.db, data.dc } }, data.magnitude };
        };
    };

    const std::string anisotropy_size_id = "n_lattice_spring_potential";
    const auto rows = parser.parse( lattice_potential_file, anisotropy_size_id, 22ul, transform_factory );
    n_pairs         = rows.size();

    data.pairs = pairfield( 0 );
    data.pairs.reserve( n_pairs );

    data.magnitudes = scalarfield( 0 );
    data.magnitudes.reserve( n_pairs );

    // use unordered_set to detect duplicate pairs
    std::unordered_set<Pair, equiv_hash<Pair>, equivalence<Pair>> seen_pairs{};

    for( auto && [pair, magnitude] : rows )
    {

        if( !seen_pairs.insert( pair ).second )
        {
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "lattice spring potential: ignoring duplicate pair: {}", pair ) );
            continue;
        }

        data.pairs.push_back( pair );
        data.magnitudes.push_back( magnitude );
    }

    data.pairs.shrink_to_fit();
    data.magnitudes.shrink_to_fit();
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read lattice spring potential from file \"{}\"", lattice_potential_file ) );
}

} // namespace

void Lattice_Spring_Potential_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    Lattice_Spring_Potential::Data & data )
{
    std::string data_file{};
    bool data_from_file = false;
    int n_pairs         = 0;

    try
    {

        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Anisotropy
        if( config_file_handle.Find( "n_lattice_spring_potential" ) )
            data_file = config_file_name;
        else if( config_file_handle.Find( "lattice_spring_potential_file" ) )
            config_file_handle >> data_file;

        if( !data_file.empty() )
        {
            // The file name should be valid so we try to read it
            Lattice_Spring_Potential_from_File( data_file, geometry, n_pairs, data );
            data_from_file = true;
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read lattice spring potential from config file \"{}\"", config_file_name ) );
    }

    if( data_from_file )
        parameter_log.emplace_back( fmt::format( "    lattice spring potential tensor from file \"{}\"", data_file ) );
}

} // namespace IO
