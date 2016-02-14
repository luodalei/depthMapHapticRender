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

#include "chai3d.h"
#include "mappingAlgorithm.h" // Header file for algorithm
#include "hapticRendering.h" // Header file for haptic rendering

//------------------------------------------------------------------------------
// DECLARED FUNCTIONS
//------------------------------------------------------------------------------

// load image to an 2D array 
int loadImage(std::string imgPath, uint* imgSize, MMatrix* depthMat);

//=====================================================================================================

int main(int argc, char* argv[])
{
	//------------------------------------------------------------------------------
	// DECLARED VARIABLES
	//------------------------------------------------------------------------------

	/* an matrix containing depth values */
	MMatrix depthMatrix(IMAGE_HEIGHT, IMAGE_WIDTH, 0.0);

	// Importing image 
	std::string imagePath0 = "../bin/resources/image/rabbit.png";
	std::string imagePath1 = "../bin/resources/image/complexScene1.png";
	std::string imagePath2 = "../bin/resources/image/ol_dm1.png"; // "museum"
	std::string imagePath3 = "../bin/resources/image/ol_dm2.png"; // "street"
	std::string imagePath4 = "../bin/resources/image/ol_dm3.png"; // "office"
	std::string imagePath5 = "../bin/resources/image/ol_dm4.png"; // "garden"
	std::string imagePath6 = "../bin/resources/image/scorpione.png"; // "scorpione"
	uint imageSize[2];

	// Load the depth matrix
	loadImage(imagePath6, imageSize, &depthMatrix);

	MMatrix mappedMatrix(IMAGE_HEIGHT, IMAGE_WIDTH, 0.0);

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
	double alpha = 2.0; // (5.0)

	mappedMatrix = basRelief(&mappedMatrix, radius2, thresh, alpha);

	//test();

	// =================== for test only : write data to .txt file (11/19/2015)
	//writeMatrix(&mappedMatrix, "modifedMap.txt");
	// =================== for test only

	//======================================================================================================

	// Rendering the image in Chai3D with haptic feedback
	hapticRender(mappedMatrix, argc, argv);

	// exit
	return (0);
}

//------------------------------------------------------------------------------

int loadImage(std::string imgPath, uint* imgSize, MMatrix* depthMat)
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
		return (-1); // Unsuccessful loading and exit
	}

	//--------------------------------------------------------------------------
	// Read depth value from each pixel of the depth image [11/13/2015 - present]
	//--------------------------------------------------------------------------
	// check whether the command is succefully executed
	bool isRight;
	//int dpVal[4];

	// depth map row (Height) and column (Width) length
	imgSize[0] = depthImage->getHeight(); // map Row Num
	imgSize[1] = depthImage->getWidth(); // map Col Num

	for (uint i = 0; i < imgSize[0]; i++)
	{
		for (uint j = 0; j < imgSize[1]; j++)
		{
			isRight = depthImage->getPixelColor(j, i, depthValue);

			if (isRight == false)
			{
				std::cout << "error - failed! [" + std::to_string(i) + "," + std::to_string(j) + "]" 
					<< std::endl;
			}

			// For gray scale image, R = G = B = GrayScale
			depthMat->setElement(i, j, depthValue.m_color[0]); // Extract R Value as depth information
		}
	}

	depthMat->div(255.0); // Normalize the depth image to [0 , 1] scale

	return (0); // Successful
}

