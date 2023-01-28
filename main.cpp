#include <array>
#include <cassert>
#include <i3d/image3d.h>
#include <iostream>
#include <limits>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <i3d/morphology.h>

constexpr std::array neighbours{std::tuple{0, 0, -1}, std::tuple{0, 0, 1},
                                std::tuple{0, -1, 0}, std::tuple{0, 1, 0},
                                std::tuple{-1, 0, 0}, std::tuple{1, 0, 0},
                                std::tuple{0, 0, 0}};

constexpr std::array forward_neigh{std::tuple{0, 0, 1}, std::tuple{0, 1, 0},
                                   std::tuple{1, 0, 0}, std::tuple{0, 0, 0}};

constexpr std::array backward_neigh{std::tuple{0, 0, -1}, std::tuple{0, -1, 0},
                                    std::tuple{-1, 0, 0}, std::tuple{0, 0, 0}};

template <typename img_t>
void inner_dil_reconstruction_3d(i3d::Image3d<img_t>& img,
                                 const i3d::Image3d<img_t>& marker) {
	i3d::Vector3d<std::size_t> size = img.GetSize();

	bool change = true;
	while (change) {
		change = false;

		// forward pass
		for (std::size_t z = 1; z < size.z - 1; ++z)
			for (std::size_t y = 1; y < size.y - 1; ++y)
				for (std::size_t x = 1; x < size.x - 1; ++x) {
					img_t max = std::numeric_limits<img_t>::min();
					for (auto [dx, dy, dz] : backward_neigh)
						max =
						    std::max(img.GetVoxel(x + dx, y + dy, z + dz), max);
					img_t new_val = std::min(max, marker.GetVoxel(x, y, z));
					change |= (img.GetVoxel(x, y, z) != new_val);
					img.SetVoxel(x, y, z, new_val);
				}

		// backward pass
		for (std::size_t z = size.z - 2; z > 0; --z)
			for (std::size_t y = size.y - 2; y > 0; --y)
				for (std::size_t x = size.x - 2; x > 0; --x) {
					img_t max = std::numeric_limits<img_t>::min();
					for (auto [dx, dy, dz] : forward_neigh)
						max =
						    std::max(img.GetVoxel(x + dx, y + dy, z + dz), max);
					img_t new_val = std::min(max, marker.GetVoxel(x, y, z));
					change |= (img.GetVoxel(x, y, z) != new_val);
					img.SetVoxel(x, y, z, new_val);
				}
	}
}

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
	for (std::size_t z = 0; z < size.z; ++z) {
		// Skip first and last iteration for 3d images
		if (size.z != 1 && (z == 0 || z == size.z - 1))
			continue;

		for (std::size_t y = 1; y < size.y - 1; ++y)
			for (std::size_t x = 1; x < size.x - 1; ++x)
				marker.SetVoxel(x, y, z, min);
	}

	i3d::Image3d<i3d::GRAY16> out = marker;

	// i3d::Reconstruction_by_dilation(marker, img, out);
	inner_dil_reconstruction_3d(out, img);

	out.SaveImage(argv[2]);
}