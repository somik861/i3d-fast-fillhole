#include <i3d/image3d.h>
#include <iostream>

int main(int argc, char** argv) {
	if (argc != 3) {
		std::cerr << "./program INPUT OUTPUT\n";
		return 1;
	}

	i3d::Image3d<i3d::GRAY16> img(argv[1]);
	i3d::Image3d<i3d::GRAY16> out = img;

	i3d::GRAY16 max = img.GetMaxValue();

	i3d::Vector3d<std::size_t> size = img.GetSize();

	// Set inner pixels to max
	for (std::size_t z = 0; z < size.z; ++z) {
		// Skip first and last iteration for 3d images
		if (size.z != 1 && (z == 0 || z == size.z - 1))
			continue;

		for (std::size_t y = 1; y < size.y - 1; ++y)
			for (std::size_t x = 1; x < size.x - 1; ++x)
				out.SetVoxel(x, y, z, max);
	}

	out.SaveImage(argv[2]);
}