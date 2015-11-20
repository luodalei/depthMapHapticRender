//==============================================================================
/*
\author    Yitian Shao
\created 11/19/2015

All the tone-mapping algorithms (as functions) are included in this file.
*/
//==============================================================================

//------------------------------------------------------------------------------
#include "mappingAlgorithm.h" // Header file for parameters
//------------------------------------------------------------------------------

// Algorithm 1 : Adjust depth intensity
int depthIntensity(float intenSacle, float depthMat[][IMAGE_WIDTH])
{
	for (int i = 0; i < IMAGE_HEIGHT; i++)
	{
		for (int j = 0; j < IMAGE_WIDTH; j++)
		{
			depthMat[i][j] = depthMat[i][j] * intenSacle;
		}
	}
	return 0;
}