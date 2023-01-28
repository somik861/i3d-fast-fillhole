#pragma once
#include <array>
#include <i3d/image3d.h>
#include <i3d/vector3d.h>
#include <tuple>
#include <type_traits>

namespace i3d {
namespace fast_morphology {
namespace details {
template <typename T, std::size_t N>
constexpr std::array<std::tuple<T, T, T>, N>
negate_coords(const std::array<std::tuple<T, T, T>, N>& arr) {
	std::array<std::tuple<T, T, T>, N> out = arr;
	for (auto& coord : out) {
		auto [x, y, z] = coord;
		coord = {-x, -y, -z};
	}
	return out;
}
template <typename T, std::size_t N>
constexpr std::array<std::tuple<T, T>, N>
negate_coords(const std::array<std::tuple<T, T>, N>& arr) {
	std::array<std::tuple<T, T>, N> out = arr;
	for (auto& coord : out) {
		auto [x, y] = coord;
		coord = {-x, -y};
	}
	return out;
}

template <typename T, std::size_t N>
constexpr std::enable_if_t<std::is_arithmetic_v<T>, std::array<T, N>>
negate_coords(const std::array<T, N>& arr) {
	std::array<T, N> out = arr;
	for (auto& coord : out)
		coord = -coord;
	return out;
}

template <typename T, std::size_t N, std::size_t M>
constexpr std::array<T, N + M> concat_arrays(const std::array<T, N>& lhs,
                                             const std::array<T, M>& rhs) {
	std::array<T, N + M> out;
	for (std::size_t i = 0; i < N; ++i)
		out[i] = lhs[i];
	for (std::size_t i = 0; i < M; ++i)
		out[i + N] = rhs[i];
	return out;
}
} // namespace details
namespace neighbour_diffs {
using t3 = std::tuple<int, int, int>;
using t2 = std::tuple<int, int>;

// ===== 3D differences
// Cell2-adj
constexpr std::array backward_3d_2{t3{0, 0, 1}, t3{0, 1, 0}, t3{1, 0, 0}};
constexpr std::array forward_3d_2 = details::negate_coords(backward_3d_2);

// Cell1-adj
constexpr std::array backward_3d_1 = details::concat_arrays(
    backward_3d_2,
    std::array{t3{0, 1, 1}, t3{1, 0, 1}, t3{1, 1, 0}, t3{-1, 1, 0},
               t3{-1, 0, 1}, t3{0, -1, 1}});
constexpr std::array forward_3d_1 = details::negate_coords(backward_3d_1);

// Cell0-adj
constexpr std::array backward_3d_0 = details::concat_arrays(
    backward_3d_1,
    std::array{t3{1, 1, 1}, t3{-1, 1, 1}, t3{1, -1, 1}, t3{-1, -1, 1}});
constexpr std::array forward_3d_0 = details::negate_coords(backward_3d_0);

// ==== 2D differences
// Cell1-adj
constexpr std::array backward_2d_1{t2{0, 1}, t2{1, 0}};
constexpr std::array forward_2d_1 = details::negate_coords(backward_2d_1);

// Cell0-adj
constexpr std::array backward_2d_0 =
    details::concat_arrays(backward_2d_1, std::array{t2{1, 1}, t2{-1, 1}});
constexpr std::array forward_2d_0 = details::negate_coords(backward_2d_0);

// ==== 1D differences
// Cell0-adj
constexpr std::array backward_1d_0{1, 0};
constexpr std::array forward_1d_0 = details::negate_coords(backward_1d_0);

} // namespace neighbour_diffs

template <typename img_t,
          typename neigh_f,
          typename mask_f,
          std::size_t N,
          std::size_t M>
void reconstruction_3d(
    Image3d<img_t>& marker,
    const Image3d<img_t>& mask,
    neigh_f neighbour_fun,
    mask_f mask_fun,
    const std::array<std::tuple<int, int, int>, N>& forward_neigh,
    const std::array<std::tuple<int, int, int>, M>& backward_neigh) {

	Vector3d<int> size = marker.GetSize();

	auto get_voxel = [size, &marker](int x, int y, int z, img_t default_) {
		if (!(0 <= x && x < size.x && 0 <= y && y < size.y && 0 <= z &&
		      z < size.z))
			return default_;
		return marker.GetVoxel(x, y, z);
	};

	bool change = true;
	while (change) {
		change = false;

		// forward pass
		for (int z = 0; z < size.z; ++z)
			for (int y = 0; y < size.y; ++y)
				for (int x = 0; x < size.x; ++x) {
					img_t val = marker.GetVoxel(x, y, z);
					for (auto [dx, dy, dz] : forward_neigh)
						val = neighbour_fun(
						    get_voxel(x + dx, y + dy, z + dz, val), val);
					img_t new_val = mask_fun(val, mask.GetVoxel(x, y, z));
					change |= (marker.GetVoxel(x, y, z) != new_val);
					marker.SetVoxel(x, y, z, new_val);
				}

		// backward pass
		for (int z = size.z - 1; z >= 0; --z)
			for (int y = size.y - 1; y >= 0; --y)
				for (int x = size.x - 1; x >= 0; --x) {
					img_t val = marker.GetVoxel(x, y, z);
					for (auto [dx, dy, dz] : backward_neigh)
						val = neighbour_fun(
						    get_voxel(x + dx, y + dy, z + dz, val), val);
					img_t new_val = mask_fun(val, mask.GetVoxel(x, y, z));
					change |= (marker.GetVoxel(x, y, z) != new_val);
					marker.SetVoxel(x, y, z, new_val);
				}
	}
}

template <typename img_t,
          typename neigh_f,
          typename mask_f,
          std::size_t N,
          std::size_t M>
void reconstruction_2d(
    Image3d<img_t>& marker,
    const Image3d<img_t>& mask,
    neigh_f neighbour_fun,
    mask_f mask_fun,
    const std::array<std::tuple<int, int>, N>& forward_neigh,
    const std::array<std::tuple<int, int>, M>& backward_neigh) {

	Vector3d<int> size = marker.GetSize();

	auto get_voxel = [size, &marker](int x, int y, int z, img_t default_) {
		if (!(0 <= x && x < size.x && 0 <= y && y < size.y && 0 <= z &&
		      z < size.z))
			return default_;
		return marker.GetVoxel(x, y, z);
	};

	bool change = true;
	while (change) {
		change = false;

		// forward pass
		for (std::size_t y = 1; y < size.y - 1; ++y)
			for (std::size_t x = 1; x < size.x - 1; ++x) {
				img_t val = marker.GetVoxel(x, y, 0);
				for (auto [dx, dy] : forward_neigh)
					val = neighbour_fun(get_voxel(x + dx, y + dy, 0, val), val);
				img_t new_val = mask_fun(val, mask.GetVoxel(x, y, 0));
				change |= (marker.GetVoxel(x, y, 0) != new_val);
				marker.SetVoxel(x, y, 0, new_val);
			}

		// backward pass
		for (std::size_t y = size.y - 2; y > 0; --y)
			for (std::size_t x = size.x - 2; x > 0; --x) {
				img_t val = marker.GetVoxel(x, y, 0);
				for (auto [dx, dy] : backward_neigh)
					val = neighbour_fun(get_voxel(x + dx, y + dy, 0, val), val);
				img_t new_val = mask_fun(val, mask.GetVoxel(x, y, 0));
				change |= (marker.GetVoxel(x, y, 0) != new_val);
				marker.SetVoxel(x, y, 0, new_val);
			}
	}
}

template <typename img_t,
          typename neigh_f,
          typename mask_f,
          std::size_t N,
          std::size_t M>
void reconstruction_1d(Image3d<img_t>& marker,
                       const Image3d<img_t>& mask,
                       neigh_f neighbour_fun,
                       mask_f mask_fun,
                       const std::array<int, N>& forward_neigh,
                       const std::array<int, M>& backward_neigh) {

	Vector3d<int> size = marker.GetSize();

	auto get_voxel = [size, &marker](int x, int y, int z, img_t default_) {
		if (!(0 <= x && x < size.x && 0 <= y && y < size.y && 0 <= z &&
		      z < size.z))
			return default_;
		return marker.GetVoxel(x, y, z);
	};

	bool change = true;
	while (change) {
		change = false;

		// forward pass
		for (std::size_t x = 1; x < size.x - 1; ++x) {
			img_t val = marker.GetVoxel(x, 0, 0);
			for (auto dx : forward_neigh)
				val = neighbour_fun(get_voxel(x + dx, 0, 0, val), val);
			img_t new_val = mask_fun(val, mask.GetVoxel(x, 0, 0));
			change |= (marker.GetVoxel(x, 0, 0) != new_val);
			marker.SetVoxel(x, 0, 0, new_val);
		}

		// backward pass
		for (std::size_t x = size.x - 2; x > 0; --x) {
			img_t val = marker.GetVoxel(x, 0, 0);
			for (auto dx : backward_neigh)
				val = neighbour_fun(get_voxel(x + dx, 0, 0, val), val);
			img_t new_val = mask_fun(val, mask.GetVoxel(x, 0, 0));
			change |= (marker.GetVoxel(x, 0, 0) != new_val);
			marker.SetVoxel(x, 0, 0, new_val);
		}
	}
}

template <typename img_t, typename neigh_f, typename mask_f>
void reconstruction(Image3d<img_t>& marker,
                    const Image3d<img_t>& mask,
                    neigh_f neighbour_fun,
                    mask_f mask_fun,
                    int cell_adjacency) {
	// 3D image
	if (marker.GetSizeZ() > 1) {
		switch (cell_adjacency) {
		case 0:
			fast_morphology::reconstruction_3d(
			    marker, mask, neighbour_fun, mask_fun,
			    fast_morphology::neighbour_diffs::forward_3d_0,
			    fast_morphology::neighbour_diffs::backward_3d_0);
			break;
		case 1:
			fast_morphology::reconstruction_3d(
			    marker, mask, neighbour_fun, mask_fun,
			    fast_morphology::neighbour_diffs::forward_3d_1,
			    fast_morphology::neighbour_diffs::backward_3d_1);
			break;
		case 2:
			fast_morphology::reconstruction_3d(
			    marker, mask, neighbour_fun, mask_fun,
			    fast_morphology::neighbour_diffs::forward_3d_2,
			    fast_morphology::neighbour_diffs::backward_3d_2);
			break;
		default:
			throw InternalException(
			    "Invalid cell neighbourhood for 3D image! (valid: {0, 1, 2})");
		}
	} else if (marker.GetSizeY() > 1) { // 2D image
		switch (cell_adjacency) {
		case 0:
			fast_morphology::reconstruction_2d(
			    marker, mask, neighbour_fun, mask_fun,
			    fast_morphology::neighbour_diffs::forward_2d_0,
			    fast_morphology::neighbour_diffs::backward_2d_0);
			break;
		case 1:
			fast_morphology::reconstruction_2d(
			    marker, mask, neighbour_fun, mask_fun,
			    fast_morphology::neighbour_diffs::forward_2d_1,
			    fast_morphology::neighbour_diffs::backward_2d_1);
			break;
		default:
			throw InternalException(
			    "Invalid cell neighbourhood for 2D image! (valid: {0, 1})");
		}
	} else if (marker.GetSizeX() > 1) { // 1D image
		if (cell_adjacency != 0)
			throw InternalException(
			    "Invalid cell neighbourhood for 1D image! (valid: {0})");

		fast_morphology::reconstruction_1d(
		    marker, mask, neighbour_fun, mask_fun,
		    fast_morphology::neighbour_diffs::forward_1d_0,
		    fast_morphology::neighbour_diffs::backward_1d_0);
	}
}

} // namespace fast_morphology

template <typename img_t>
void Reconstruction_by_dilation_fast(const i3d::Image3d<img_t>& marker,
                                     const i3d::Image3d<img_t>& mask,
                                     i3d::Image3d<img_t>& out,
                                     int cell_adjacency = 0) {
	if (marker.GetSize() != mask.GetSize())
		throw InternalException("Mask and marker must be the same size");
	out = marker;

	fast_morphology::reconstruction(
	    out, mask, [](img_t a, img_t b) { return std::max(a, b); },
	    [](img_t a, img_t b) { return std::min(a, b); }, cell_adjacency);
}

template <typename img_t>
void Reconstruction_by_erosion_fast(const i3d::Image3d<img_t>& marker,
                                    const i3d::Image3d<img_t>& mask,
                                    i3d::Image3d<img_t>& out,
                                    int cell_adjacency = 0) {
	if (marker.GetSize() != mask.GetSize())
		throw InternalException("Mask and marker must be the same size");
	out = marker;

	fast_morphology::reconstruction(
	    out, mask, [](img_t a, img_t b) { return std::min(a, b); },
	    [](img_t a, img_t b) { return std::max(a, b); }, cell_adjacency);
}
} // namespace i3d