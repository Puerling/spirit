#pragma once

#include <io/VTK_Geometry.hpp>

namespace IO
{

namespace HDF5
{

void write_fields(
    const std::string & outfile, const VTK::UnstructuredGrid & geometry,
    const std::vector<VTK::FieldDescriptor> & fields );

#ifndef SPIRIT_USE_HDF5
inline void
write_fields( const std::string & outfile, const VTK::UnstructuredGrid &, const std::vector<VTK::FieldDescriptor> & )
{
    spirit_throw(
        Utility::Exception_Classifier::Not_Implemented, Utility::Log_Level::Error,
        fmt::format( "Cannot write VTKHDF file. This build was compiled without HDF5 support. (\"{}\")", outfile ) );
}
#endif

} // namespace HDF5

} // namespace IO
