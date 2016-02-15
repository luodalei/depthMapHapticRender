//==============================================================================
/*
\author    Yitian Shao
\created 11/20/2015
\updated 02/15/2015

All the tone-mapping algorithms (as functions) are included in this file.
*/
//==============================================================================

#pragma once // Ensure unique inclusion

//------------------------------------------------------------------------------
#include <string>
#include "poissonSolver.h" 
#include "mathTool.h" 
//------------------------------------------------------------------------------

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
// FOR TEST ONLY
//------------------------------------------------------------------------------
void test();