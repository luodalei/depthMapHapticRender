//==============================================================================
/*
\author    Yitian Shao
\created 11/20/2015

All the tone-mapping algorithms (as functions) are included in this file.
*/
//==============================================================================

#include <string>
#include "mathTool.h"
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// MARCO
//------------------------------------------------------------------------------

// The width of the imported depth image is fixed to be 960 pixels
#ifndef IMAGE_WIDTH
#  define IMAGE_WIDTH 960
#endif

// The height of the imported depth image is fixed to be 540 pixels
#ifndef IMAGE_HEIGHT
#  define IMAGE_HEIGHT 540
#endif

// Pi
#ifndef M_PI
#  define M_PI 3.1415926535897
#endif

// Define dype
typedef std::tuple<uint, int, uint, int> Range2D; // Store range of rows and columns of a matrix
typedef std::tuple<MMatrix, MMatrix, MMatrix> M3MatPtr; // 3 MMatrix tuple


//------------------------------------------------------------------------------
// DECLARED FUNCTIONS
//------------------------------------------------------------------------------

// Algorithm 1 : Adjust depth intensity and apply Gaussian filter
MMatrix gaussian(double intenSacle, MMatrix* depthMat, uint radius, int sigma);

// Algorithm 2 : Bas-Relief -> Gradient compression
MMatrix basRelief(MMatrix* depthMat, uint radius, double thres, double alpha);

// Edge detection (matrix differentiation)
M3MatPtr matrixDiff(MMatrix* depthMat, MMatrix ker, bool isDirect);

//------------------------------------------------------------------------------
// ASSISTIVE FUNCTIONS
//------------------------------------------------------------------------------
/* Import the original depth map image */
void readMatrix(MMatrix* mat, std::string filepath);

/* Export the mapped image to a .txt file */
void writeMatrix(MMatrix* mat, std::string filename);

//------------------------------------------------------------------------------
// FOR TEST ONLY
//------------------------------------------------------------------------------
void test();