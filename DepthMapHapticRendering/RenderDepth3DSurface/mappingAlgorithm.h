//==============================================================================
/*
\author    Yitian Shao
\created 11/20/2015

All the tone-mapping algorithms (as functions) are included in this file.
*/
//==============================================================================

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

// Algorithm 1 : Adjust depth intensity
int depthIntensity(float intenSacle, float depthMat[][IMAGE_WIDTH]);