#pragma once

#include <data/Geometry.hpp>
#include <data/Misc.hpp>
#include <engine/Span.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <utility/Exception.hpp>

#include <utility>

namespace IO
{

namespace VTK
{

using FieldDescriptor = std::pair<std::string_view, const vectorfield *>;

class UnstructuredGrid
{
public:
    UnstructuredGrid( const ::Data::Geometry & geometry ) : m_geometry( geometry )
    {
        m_is_3d  = this->geometry().nos > 1 && !tetrahedra().empty();
        int step = 0;

        if( m_is_3d )
        {
            step    = 4;
            m_types = std::vector<std::uint8_t>( tetrahedra().size(), 10 );
        }
        else
        {
            step    = 3;
            m_types = std::vector<std::uint8_t>( triangles().size(), 5 );
        }

        m_offsets = [n = m_types.size() + 1, step]
        {
            auto offsets = std::vector<int>( n, 0 );
            std::generate(
                offsets.begin(), offsets.end(),
                [step = std::as_const( step ), i = 0]() mutable { return step * ( i++ ); } );
            return offsets;
        }();
    };

    auto geometry() const -> const ::Data::Geometry &
    {
        return m_geometry;
    }

    auto positions() const -> const vectorfield &
    {
        return geometry().positions;
    }

    auto tetrahedra() const -> const std::vector<::Data::tetrahedron_t> &
    {
        return geometry().tetrahedra();
    }

    auto triangles() const -> const std::vector<::Data::triangle_t> &
    {
        return geometry().triangulation();
    }

    auto cell_size() const -> std::size_t
    {
        return m_types.size();
    }

    auto connectivity() const -> Engine::Span<const int>
    {
        if( m_is_3d )
        {
            if( tetrahedra().empty() )
                return {};

            auto & first = tetrahedra().front();
            return Engine::Span( first.data(), first.size() * tetrahedra().size() );
        }
        else
        {
            if( triangles().empty() )
                return {};

            auto & first = triangles().front();
            return Engine::Span( first.data(), first.size() * triangles().size() );
        }
    }

    auto offsets() const -> const std::vector<int> &
    {
        return m_offsets;
    };

    auto types() const -> const std::vector<std::uint8_t> &
    {
        return m_types;
    };

private:
    ::Data::Geometry m_geometry;

    bool m_is_3d = false;
    std::vector<std::uint8_t> m_types{};
    std::vector<int> m_offsets{};
};

} // namespace VTK

} // namespace IO
