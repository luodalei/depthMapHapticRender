//==============================================================================
/*
\author    Yitian Shao
\templete from <http://www.chai3d.org>
\created 02/15/2016 v1.02
This is the third step on realizing depth map haptic rendering.
Bas-relief using gradient compression and Multigrid method succeed.
Image processing and haptic rendering programs are separated.

This file contains functions needed to solve the poisson equation, which can be 
used to perform 2D matrix integration.

Poisson equation

*/
//==============================================================================

//------------------------------------------------------------------------------
#include "poissonSolver.h" 
//------------------------------------------------------------------------------

#include <time.h> // Evaluate algorithm by time

//------------------------------------------------------------------------------

// double breakT = 300.0; // Maximum runing time (sec)
MVector err1(2); // Recording conversion errors (Debug only)

//------------------------------------------------------------------------------

/* Solving Poisson equation: \triangledown^2 V = \rho */
/* Multigrid Method */
MMatrix IntgralSolver(MMatrix* V1, MMatrix* rho1, double accuracy, uint soomthNum)
{
	uint height = V1->getRowsNum();
	uint width = V1->getColsNum();

	// Wrap the matrix (V1) with a square matrix (sV) with length equal to power of 2
	uint mLength = (height > width) ? height : width;
	uint powTwo = 1;

	double stepLen = 1.0; // Step length of poisson solver

	while (powTwo + 2 < mLength) // Number of interior points + two outliers
	{
		powTwo *= 2;
	}
	mLength = powTwo + 2;
	//std::cout << "mLength = " << mLength << std::endl;

	MMatrix sV(mLength, mLength, 0.0); // Squared V matrix
	MMatrix sRho(mLength, mLength, 0.0); // Squared rho matrix

										 // Fill the center part of the square matrix with the input matrix
	uint hShift = (mLength - height) / 2;
	uint wShift = (mLength - width) / 2;

	for (uint i = 0; i < height; i++)
	{
		for (uint n = 0; n <= wShift; n++) // Fill left extra block
		{
			sV.setElement(hShift + i, n, V1->getElement(i, 0));
		}
		for (uint n = wShift + width - 1; n < mLength; n++) // Fill right extra block
		{
			sV.setElement(hShift + i, n, V1->getElement(i, width - 1));
		}

	}

	for (uint j = 0; j < width; j++)
	{
		for (uint m = 0; m <= hShift; m++) // Fill top extra block
		{
			sV.setElement(m, wShift + j, V1->getElement(0, j));
		}
		for (uint m = hShift + height - 1; m < mLength; m++) // Fill bottom extra block
		{
			sV.setElement(m, wShift + j, V1->getElement(height - 1, j));
		}
	}

	// Extend rho to square matrix
	sRho.copyBlock(*rho1, height, width, std::make_tuple(0, 0, hShift, wShift));

	// Fill blocks at four corners
	sV.setBlock(V1->getElement(0, 0), std::make_tuple(0, hShift, 0, wShift));
	sV.setBlock(V1->getElement(height - 1, 0), std::make_tuple(hShift + height - 1, mLength - 1, 0, wShift));
	sV.setBlock(V1->getElement(0, width - 1), std::make_tuple(0, hShift, wShift + width - 1, mLength - 1));
	sV.setBlock(V1->getElement(height - 1, 0), std::make_tuple(hShift + height - 1, mLength - 1,
		wShift + width - 1, mLength - 1));

	MMatrix sV_new = sV; // Update matrix

	clock_t t0 = clock(); // Initial time of the solver

	int steps = 0; //  count iteration steps

				   //double omega = 2 / (1 + M_PI / sqrt(height*width)); // For SOR method only
	double omega = 1.0; // For Multigrid method method 

	bool continueItr = true; // whether the iteration continues

	std::cout << "Solving Equation ..." << std::endl;

	double errRecord = 1; // To prevent error increase

						  /* Start Iteration */
	while (continueItr)
	{
		sV = sV_new;

		/*  Recursion of the multigrid method  */
		twoGrid(&soomthNum, &sV_new, &sRho, &omega, stepLen);

		double error = 0;
		uint n = 0;

		// Compute error ( relative: (1-sV) / sV_new )
		for (uint i = 1; i <= mLength - 2; i++)
		{
			for (uint j = 1; j <= mLength - 2; j++)
			{
				double oldVal = sV.getElement(i, j);
				double newVal = sV_new.getElement(i, j);
				if (newVal != 0)
				{
					if (newVal != oldVal)
					{
						error += abs(1 - oldVal / newVal);
						n++;
					}
				}
			}
		}
		// std::cout << error << " , n = " << n << std::endl;

		if (n != 0) error /= n;

		// Debug only --------------------------------------
		err1.append(error);
		err1.append(double(clock() - t0) / CLOCKS_PER_SEC);
		// -------------------------------------------------

		if ((steps > 1) && (error < accuracy))
		{
			continueItr = false;
		}

		//if (breakT <= double(clock() - t0) / CLOCKS_PER_SEC) // 'Break' if exceed limited time
		//{
		//	continueItr = false;
		//}

		steps += ((2 * soomthNum + 1) * log2(powTwo));
	}

	std::cout << "Number of steps = " << steps << std::endl;
	std::cout << "CPU time = " << double(clock() - t0) / CLOCKS_PER_SEC << " sec" << std::endl;

	// Crop the resulted square matrix to original size and discard extended parts.
	MMatrix retMat(height, width, 0.0);
	retMat.copyBlock(sV_new, height, width, std::make_tuple(hShift, wShift, 0, 0));

	writeMatrix(&err1, "err1.txt"); // Debug only

	return retMat;
}

/* Gauss-Seidel method */
void Gauss_Seidel(MMatrix* u1, MMatrix* r1)
{
	uint height = u1->getRowsNum();
	uint width = u1->getColsNum();

	for (uint i = 1; i <= height - 2; i++)
	{
		for (uint j = 1; j <= width - 2; j++)
		{
			u1->setElement(i, j, 0.25*(u1->getElement(i - 1, j)
				+ u1->getElement(i + 1, j)
				+ u1->getElement(i, j - 1)
				+ u1->getElement(i, j + 1) - r1->getElement(i, j)));
		}
	}
}

/* Successive Over Relaxation (SOR) method */
void SOR(double* omega, MMatrix* u1, MMatrix* r1, double h)
{
	uint height = u1->getRowsNum();
	uint width = u1->getColsNum();

	for (uint i = 1; i <= height - 2; i++) // Interior points only
	{
		for (uint j = 1; j <= width - 2; j++) // Interior points only
		{
			if ((i + j) % 2 == 0) // Update even sites
			{
				u1->setElement(i, j, (1 - *omega) * u1->getElement(i, j)
					+ *omega * 0.25 * (u1->getElement(i - 1, j) + u1->getElement(i + 1, j)
						+ u1->getElement(i, j - 1) + u1->getElement(i, j + 1)
						+ h * h * r1->getElement(i, j)));
			}
		}
	}

	for (uint i = 1; i <= height - 2; i++) // Interior points only
	{
		for (uint j = 1; j <= width - 2; j++) // Interior points only
		{
			if ((i + j) % 2 == 1) // Update odd sites
				u1->setElement(i, j, (1 - *omega) * u1->getElement(i, j)
					+ *omega * 0.25 * (u1->getElement(i - 1, j) + u1->getElement(i + 1, j)
						+ u1->getElement(i, j - 1) + u1->getElement(i, j + 1)
						+ h * h * r1->getElement(i, j)));
		}
	}
}

/*  Subroutine of recursion of the multigrid method  */
void twoGrid(uint* smoothN, MMatrix* u1, MMatrix* r1, double* omega, double h)
{
	// Length of current square matrix (fine grid) containing interior points only
	uint inLength = u1->getRowsNum() - 2;

	// State when only one interior point left (+ two outliers)
	if (inLength == 1)
	{
		u1->setElement(1, 1, 0.25 * (u1->getElement(0, 1) + u1->getElement(2, 1)
			+ u1->getElement(1, 0) + u1->getElement(1, 2) + h * h * r1->getElement(1, 1)));
		return; // Going back to call function
	}

	// Pre-smoothing using SOR method
	for (uint i = 0; i < *smoothN; i++) { SOR(omega, u1, r1, h); }

	// Compute the residual (fine grid)
	MMatrix fineGrid(inLength + 2, inLength + 2, 0.0);  // Number of interior points + two outliers	
	for (uint i = 1; i <= inLength; i++) // Interior points only
	{
		for (uint j = 1; j <= inLength; j++) // Interior points only
		{
			fineGrid.setElement(i, j, (u1->getElement(i + 1, j) + u1->getElement(i - 1, j)
				+ u1->getElement(i, j + 1) + u1->getElement(i, j - 1)
				- 4 * u1->getElement(i, j)) / (h * h) + r1->getElement(i, j));
		}
	}

	// Length of coarse grid = half of the length of fine grid
	uint coarLength = inLength / 2;

	// Compute the residual (coarse grid)
	MMatrix coarseGrid(coarLength + 2, coarLength + 2, 0.0);  // Number of coarse points + two outliers	
	for (uint m = 1; m <= coarLength; m++) // Coarse points only
	{
		uint i = 2 * m - 1;
		for (uint n = 1; n <= coarLength; n++) // Coarse points only
		{
			uint j = 2 * n - 1;
			coarseGrid.setElement(m, n, 0.25 * (fineGrid.getElement(i, j)
				+ fineGrid.getElement(i + 1, j) + fineGrid.getElement(i, j + 1)
				+ fineGrid.getElement(i + 1, j + 1)));
		}
	}

	// Initialize a correction matrix on coarse grid
	MMatrix correction(coarLength + 2, coarLength + 2);

	// ---------------------------------- Going in -----------------------------------

	// Recursion
	twoGrid(smoothN, &correction, &coarseGrid, omega, (2 * h));

	// ---------------------------------- Going out ----------------------------------

	// Prolongate correction (coarse) to fine grid 	
	for (uint m = 1; m <= coarLength; m++) // Coarse points only
	{
		uint i = 2 * m - 1;
		for (uint n = 1; n <= coarLength; n++) // Coarse points only
		{
			uint j = 2 * n - 1;
			fineGrid.setElement(i, j, correction.getElement(m, n));
			fineGrid.setElement(i + 1, j, correction.getElement(m, n));
			fineGrid.setElement(i, j + 1, correction.getElement(m, n));
			fineGrid.setElement(i + 1, j + 1, correction.getElement(m, n));
		}
	}

	// Correct u1
	(*u1) += fineGrid;

	// Post-smoothing using SOR method
	for (uint i = 0; i < *smoothN; i++) { SOR(omega, u1, r1, h); }
}