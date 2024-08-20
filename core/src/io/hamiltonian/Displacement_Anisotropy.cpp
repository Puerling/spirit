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

using Engine::SpinLattice::Interaction::Displacement_Anisotropy;

void Displacement_Anisotropy_from_File(
    const std::string & displacement_anisotropy_file, const Data::Geometry &, int & n_pairs,
    Displacement_Anisotropy::Data & data )
try
{
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading displacement anisotropy from file {}", displacement_anisotropy_file ) );

    // parser initialization
    using LatticePotentialTableParser = TableParser<int, int, int, int, int, scalar>;
    const LatticePotentialTableParser parser( { "i", "j", "da", "db", "dc", "k" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&displacement_anisotropy_file]( const std::map<std::string_view, int> & idx )
    {
        const bool valid_pairs = idx.at( "i" ) >= 0 && idx.at( "j" ) >= 0 && idx.at( "da" ) >= 0 && idx.at( "db" ) >= 0
                                 && idx.at( "dc" ) >= 0;

        if( !valid_pairs || idx.at( "k" ) < 0 )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "No displacement anisotropy data could be found in header of file \"{}\"",
                     displacement_anisotropy_file ) );

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

    const std::string size_id = "n_displacement_anisotropy";
    const auto rows           = parser.parse( displacement_anisotropy_file, size_id, 10ul, transform_factory );
    n_pairs                   = rows.size();

    data.pairs = pairfield( 0 );
    data.pairs.reserve( n_pairs );

    data.magnitudes = scalarfield( 0 );
    data.magnitudes.reserve( n_pairs );

    // use unordered_set to detect duplicate pairs (in this case there are no equivalent pairs)
    std::unordered_set<Pair, std::hash<Pair>, std::equal_to<Pair>> seen_pairs{};

    for( auto && [pair, magnitude] : rows )
    {
        if( !seen_pairs.insert( pair ).second )
        {
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "displacement anisotropy: ignoring duplicate pair: {}", pair ) );
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
    spirit_rethrow(
        fmt::format( "Could not read displacement anisotropy from file \"{}\"", displacement_anisotropy_file ) );
}

} // namespace

void Displacement_Anisotropy_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    Displacement_Anisotropy::Data & data )
{
    std::string data_file{};
    bool data_from_file = false;
    int n_pairs         = 0;

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Anisotropy
        if( config_file_handle.Find( "n_displacement_anisotropy" ) )
            data_file = config_file_name;
        else if( config_file_handle.Find( "displacement_anisotropy_file" ) )
            config_file_handle >> data_file;

        if( !data_file.empty() )
        {
            // The file name should be valid so we try to read it
            Displacement_Anisotropy_from_File( data_file, geometry, n_pairs, data );
            data_from_file = true;
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read displacement anisotropy from config file \"{}\"", config_file_name ) );
    }

    if( data_from_file )
        parameter_log.emplace_back( fmt::format( "    lattice displacement anisotropy from file \"{}\"", data_file ) );
}

} // namespace IO
