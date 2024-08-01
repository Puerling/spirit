#ifdef SPIRIT_USE_HDF5
#include <H5Cpp.h>

#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <io/HDF5_File.hpp>
#include <utility/Logging.hpp>

using Utility::Log_Level, Utility::Log_Sender, Utility::Exception_Classifier;

namespace IO
{

namespace HDF5
{

namespace
{

template<typename T>
struct h5_type
{
    static const H5::PredType & value;
};

template<typename T>
static const H5::PredType h5_type_v = h5_type<T>::value;

// clang-format off
template<> [[maybe_unused]] const H5::PredType& h5_type<std::uint8_t>::value = H5::PredType::NATIVE_UINT8;
template<> [[maybe_unused]] const H5::PredType& h5_type<unsigned int>::value = H5::PredType::NATIVE_UINT;
template<> [[maybe_unused]] const H5::PredType& h5_type<int>::value          = H5::PredType::NATIVE_INT;
template<> [[maybe_unused]] const H5::PredType& h5_type<float>::value        = H5::PredType::NATIVE_FLOAT;
template<> [[maybe_unused]] const H5::PredType& h5_type<double>::value       = H5::PredType::NATIVE_DOUBLE;
// clang-format on

template<typename T>
void write_scalar( H5::Group & root, const char * label, const T value )
{
    static constexpr hsize_t size = 1;

    auto node = root.createDataSet( label, h5_type_v<T>, H5::DataSpace( 1, &size ) );
    node.write( &value, h5_type_v<T> );
};

template<typename T, std::size_t N>
void write_array( H5::Group & root, const char * label, const std::array<hsize_t, N> & dim, const T * data )
{
    auto node = root.createDataSet( label, h5_type_v<T>, H5::DataSpace( N, dim.data() ) );
    node.write( data, h5_type_v<T> );
};

template<typename... T>
constexpr auto dim_array( T... args ) -> std::array<hsize_t, sizeof...( T )>
{
    return std::array<hsize_t, sizeof...( args )>{ static_cast<hsize_t>( args )... };
}

template<typename Iterable>
void write_vectorfields( H5::Group & group, const std::size_t ref_size, const Iterable & fields )
{
    for( const auto & [name, data] : fields )
    {
        if( data == nullptr )
            continue;

        if( data->size() != ref_size )
        {
            Log( Log_Level::Warning, Log_Sender::IO, "Skipping vectorfield with invalid size!" );
            continue;
        }
        write_array<scalar>( group, name.data(), dim_array( data->size(), 3 ), data->front().data() );
    }
}

} // namespace

void write_fields(
    const std::string & outfile, const VTK::UnstructuredGrid & geometry,
    const std::vector<VTK::FieldDescriptor> & fields )
{
    static constexpr std::string_view grid_type = "UnstructuredGrid";
    static constexpr std::array<int, 2> version = { 2, 2 };

    try
    {
        H5::H5File hdf_file( outfile, H5F_ACC_TRUNC );
        auto root = hdf_file.createGroup( "/VTKHDF" );

        // Version
        std::array<hsize_t, 1> version_dim{ version.size() };
        auto attr_version = root.createAttribute(
            "Version", h5_type_v<int>, H5::DataSpace( version_dim.size(), version_dim.data() ) );
        attr_version.write( h5_type_v<int>, version.data() );

        // Grid Type
        H5::StrType str_type( H5::PredType::C_S1, grid_type.length() );
        auto attr_type = root.createAttribute( "Type", str_type, H5::DataSpace( H5S_SCALAR ) );
        attr_type.write( str_type, grid_type.data() );

        write_scalar<int>( root, "NumberOfPoints", geometry.positions().size() );
        write_scalar<int>( root, "NumberOfCells", geometry.cell_size() );
        write_scalar<int>( root, "NumberOfConnectivityIds", geometry.connectivity().size() );

        write_array<scalar>(
            root, "Points", dim_array( geometry.positions().size(), 3 ), geometry.positions().front().data() );

        write_array<int>(
            root, "Connectivity", dim_array( geometry.connectivity().size() ), geometry.connectivity().data() );

        write_array<int>( root, "Offsets", dim_array( geometry.offsets().size() ), geometry.offsets().data() );

        write_array<std::uint8_t>( root, "Types", dim_array( geometry.types().size() ), geometry.types().data() );

        if( !fields.empty() )
        {
            auto point_data_group = root.createGroup( "PointData" );
            write_vectorfields( point_data_group, geometry.positions().size(), fields );
        }
    }
    catch( H5::FileIException & error )
    {
        error.printErrorStack();
        spirit_throw( Exception_Classifier::Unknown_Exception, Log_Level::Error, "Error creating HDF5 file" );
    }
    catch( H5::DataSetIException & error )
    {
        error.printErrorStack();
        spirit_throw( Exception_Classifier::Unknown_Exception, Log_Level::Error, "Error creating HDF5 dataset" );
    }
    catch( H5::DataSpaceIException & error )
    {
        error.printErrorStack();
        spirit_throw( Exception_Classifier::Unknown_Exception, Log_Level::Error, "Error creating HDF5 dataspace" );
    }
    catch( H5::AttributeIException & error )
    {
        error.printErrorStack();
        spirit_throw( Exception_Classifier::Unknown_Exception, Log_Level::Error, "Error creating HDF5 attribute" );
    }
};

} // namespace HDF5

} // namespace IO
#endif
