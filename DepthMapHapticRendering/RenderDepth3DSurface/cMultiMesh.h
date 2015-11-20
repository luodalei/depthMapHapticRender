//==============================================================================
/*
	\author    Yitian Shao
	\created 11/19/2015

	Header file for "main_cMultiMesh.cpp"
	This is the second step on realizing depth map haptic rendering.
	The goal here is to import a depth map image in format like '.png' and
	construct a 3D surface through a mesh. Then rendering it.
*/
//==============================================================================

// The width of the imported depth image is fixed to be 960 pixels
#ifndef IMAGE_WIDTH
#  define IMAGE_WIDTH 960
#endif

// The height of the imported depth image is fixed to be 540 pixels
#ifndef IMAGE_HEIGHT
#  define IMAGE_HEIGHT 540
#endif