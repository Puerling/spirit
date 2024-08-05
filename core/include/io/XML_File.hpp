#pragma once

#include <io/Fileformat.hpp>
#include <io/VTK_Geometry.hpp>

namespace IO
{

namespace XML
{

void write_fields(
    const std::string & outfile, const VTK::UnstructuredGrid & geometry, const VF_FileFormat format,
    const std::vector<VTK::FieldDescriptor> & fields );

} // namespace XML

} // namespace IO
