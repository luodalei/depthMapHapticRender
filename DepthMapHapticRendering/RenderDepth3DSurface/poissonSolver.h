//==============================================================================
/*
\author    Yitian Shao
\created 02/15/2016

All the declaration poisson solver functions are included in this file.
*/
//==============================================================================

#pragma once // Ensure unique inclusion

//------------------------------------------------------------------------------
#include "mathTool.h"
//------------------------------------------------------------------------------

/* Solving Poisson equation to estimate 2D integrals*/
MMatrix IntgralSolver(MMatrix* V1, MMatrix* rho1, double accuracy, uint soomthNum);

/* Gauss-Seidel method */
void Gauss_Seidel(MMatrix* u1, MMatrix* r1);

/* Successive Over Relaxation (SOR) method */
void SOR(double* omega, MMatrix* u1_new, MMatrix* r1, double h);

/*  Subroutine of recursion of the multigrid method  */
void twoGrid(uint* smthNum, MMatrix* u1, MMatrix* r1, double* omega, double h);