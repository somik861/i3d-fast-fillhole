#include <cassert>
#include <i3d/image3d.h>
#include <i3d/morphology.h>
#include <iostream>
#include <limits>
#include <queue>
#include "fast_morphology.hpp"

int main(int argc, char** argv) {
	if (argc != 3) {
		std::cerr << "./program INPUT OUTPUT\n";
		return 1;
	}

	i3d::Image3d<i3d::GRAY16> img(argv[1]);
	i3d::Image3d<i3d::GRAY16> marker = img;

	i3d::GRAY16 min = img.GetMinValue();

	i3d::Vector3d<std::size_t> size = img.GetSize();

	// Set inner pixels to max
	for (std::size_t z = 1; z < size.z - 1; ++z) {
		for (std::size_t y = 1; y < size.y - 1; ++y)
			for (std::size_t x = 1; x < size.x - 1; ++x)
				marker.SetVoxel(x, y, z, min);
	}

	i3d::Image3d<i3d::GRAY16> out = marker;

	i3d::Reconstruction_by_dilation_fast(marker, img, out, 2);
	// i3d::Reconstruction_by_dilation(marker, img, out);

	out.SaveImage(argv[2]);
}