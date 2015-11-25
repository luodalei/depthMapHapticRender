//==============================================================================
/*
\author    Yitian Shao
\created 11/20/2015

All the tone-mapping algorithms (as functions) are included in this file.
*/
//==============================================================================

#include <string>
//------------------------------------------------------------------------------
using namespace std;

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


//------------------------------------------------------------------------------
// DECLARED FUNCTIONS
//------------------------------------------------------------------------------

// Algorithm 1 : Adjust depth intensity and apply Gaussian filter
double** gaussian(double intenSacle, double** depthMat, int radius, int sigma);

//------------------------------------------------------------------------------
// ASSISTIVE FUNCTIONS
//------------------------------------------------------------------------------
/* Import the original depth map image */
void readMatrix(double** mat, int width, int height, string filepath);

/* Export the mapped image to a .txt file */
void writeMatrix(double** mat, int width, int height, string filename);