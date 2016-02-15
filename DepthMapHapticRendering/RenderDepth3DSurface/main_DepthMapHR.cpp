//==============================================================================
/*
\author    Yitian Shao
\templete from <http://www.chai3d.org>
\created 02/14/2016 v1.02
This is the third step on realizing depth map haptic rendering.
Bas-relief using gradient compression and Multigrid method succeed. 
Image processing and haptic rendering programs are separated. 
*/
//==============================================================================

#include "mappingAlgorithm.h" // Header file for algorithm
#include "hapticRendering.h" // Header file for haptic rendering
/* ("chai3d.h" is included in "hapticRendering.h") */

///////////////////////////////////////////////////////////////////////////////
// DECLARED FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

// load image to an 2D array 
MMatrix loadImage(std::string imgPath);

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLE
///////////////////////////////////////////////////////////////////////////////

//=====================================================================================================

int main(int argc, char* argv[])
{
	//------------------------------------------------------------------------------
	// DECLARED VARIABLES
	//------------------------------------------------------------------------------

	// Importing image 
	std::string imagePath0 = "../bin/resources/image/rabbit.png";
	std::string imagePath1 = "../bin/resources/image/complexScene1.png";
	std::string imagePath2 = "../bin/resources/image/ol_dm1.png"; // "museum"
	std::string imagePath3 = "../bin/resources/image/ol_dm2.png"; // "street"
	std::string imagePath4 = "../bin/resources/image/ol_dm3.png"; // "office"
	std::string imagePath5 = "../bin/resources/image/ol_dm4.png"; // "garden"
	std::string imagePath6 = "../bin/resources/image/scorpione.png"; // "scorpione"
	//std::string imagePath7 = "../bin/resources/image/large_scene1.png"; // "large scene (1920 * 1080)"
	//std::string imagePath7 = "../bin/resources/image/large_scene2.png"; // "large scene (1600 * 900)"	

	// Load the depth matrix
	MMatrix depthMatrix = loadImage(imagePath5);

	MMatrix mappedMatrix(depthMatrix.getRowsNum(), depthMatrix.getColsNum(), 0.0);

	mappedMatrix = depthMatrix; // Original (no filter) 
	///////////////////////////////////////////////////////////////////////////
	// Apply algorithm to the depth map
	///////////////////////////////////////////////////////////////////////////

	// 1. Gaussian filtering (Optional)
	//uint radius = 2; // (5) changed 01 / 15 / 2016
	//int sigma = 4; // (4)
	//mappedMatrix = gaussian(0.5, &depthMatrix, radius, sigma); // Gaussian filter 

	// 2. Gradient magnitude compression and bas relief
	uint radius2 = 2; // (2)
	double thresh = 0.01; // (0.01)
	double alpha = 2.0; // (5.0)(2.0)

	//mappedMatrix = basRelief(&mappedMatrix, radius2, thresh, alpha);

	//test();

	// =================== for test only : write data to .txt file (11/19/2015)
	writeMatrix(&mappedMatrix, "modifedMap.txt");
	// =================== for test only

	//======================================================================================================

	// Rendering the image in Chai3D with haptic feedback
	hapticRender(mappedMatrix, argc, argv);

	// exit
	return (0);
}

//------------------------------------------------------------------------------

MMatrix loadImage(std::string imgPath)
{
	//--------------------------------------------------------------------------
	// LOAD FILE 
	// Created on 11/13/2015
	// Updated on 02/14/2016
	//--------------------------------------------------------------------------

	/*11/13/2015 a depth image*/
	chai3d::cImage* depthImage = new chai3d::cImage(800, 500, GL_RGB, GL_UNSIGNED_INT);

	/*11/13/2015 a depth value*/
	chai3d::cColorb depthValue;

	// check whether file loaded or not
	bool fileload;

	// create a new image
	//depthImage = new cImage(windowW, windowH, GL_RGB, GL_UNSIGNED_INT);

	// import the depth image from a .png file
	fileload = depthImage->loadFromFile(imgPath);
	if (!fileload)
	{
		std::cout << "Error - Image failed to load correctly." << std::endl;
		MMatrix errMat(1, 1, -1.0);
		return errMat; // Unsuccessful loading and exit
	}

	//--------------------------------------------------------------------------
	// Read depth value from each pixel of the depth image [11/13/2015 - present]
	//--------------------------------------------------------------------------
	// check whether the command is succefully executed
	bool isRight;
	//int dpVal[4];

	// depth map row (Height) and column (Width) length
	size_t height = depthImage->getHeight(); // map Row Num
	size_t width = depthImage->getWidth(); // map Col Num

	MMatrix depthMat(height, width, 0.0);

	for (uint i = 0; i < height; i++)
	{
		for (uint j = 0; j < width; j++)
		{
			isRight = depthImage->getPixelColor(j, i, depthValue);

			if (isRight == false)
			{
				std::cout << "error - failed! [" + std::to_string(i) + "," + std::to_string(j) + "]" 
					<< std::endl;
			}

			// For gray scale image, R = G = B = GrayScale
			depthMat.setElement(i, j, depthValue.m_color[0]); // Extract R Value as depth information
		}
	}

	std::cout << "Image with size of " << height << " - by - " << width 
			  << " has been loaded successfully" << std::endl;

	depthMat.div(255.0); // Normalize the depth image to [0 , 1] scale

	return depthMat; 
}

