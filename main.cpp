#include <i3d/image3d.h>
#include <iostream>

int main(int argc, char** argv) {
	if (argc != 3) {
		std::cerr << "./program INPUT OUTPUT\n";
		return 1;
	}

	i3d::Image3d<i3d::GRAY16> img(argv[1]);

	img.SaveImage(argv[2]);
}