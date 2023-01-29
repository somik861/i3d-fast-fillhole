#pragma once
#include <i3d/image3d.h>

namespace i3d {
template <typename img_t>
void Reconstruction_by_dilation_fast(const i3d::Image3d<img_t>& marker,
                                     const i3d::Image3d<img_t>& mask,
                                     i3d::Image3d<img_t>& out,
                                     int cell_adjacency = 0 );

template <typename img_t>
void Reconstruction_by_erosion_fast(const i3d::Image3d<img_t>& marker,
                                    const i3d::Image3d<img_t>& mask,
                                    i3d::Image3d<img_t>& out,
                                    int cell_adjacency = 0);
}

#include "_fast_morphology_impl.hpp"