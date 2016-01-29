//==============================================================================
/*
\author    Yitian Shao
\created 11/19/2015

All the tone-mapping algorithms (as functions) are included in this file.
Note that some functions take 'MMatrix' as input while others take 'MMatrix * ' 
instead.
*/
//==============================================================================

//------------------------------------------------------------------------------
#include "mappingAlgorithm.h" // Header file for parameters
//------------------------------------------------------------------------------

#include <iostream> 
#include <fstream> 
#include <string>
#include<tuple>
#include <time.h> // Evaluate algorithm by time

//------------------------------------------------------------------------------

///////////////////////////////////////////////////////////////////////////////
// FUNCTION (LOCAL USE ONLY) DECLARATION 
///////////////////////////////////////////////////////////////////////////////

/* 2D Filter */
MMatrix filter(MMatrix* mat, MMatrix ker, Range2D filtRange = std::make_tuple(0, 0, 0, 0));

/* Construct gaussian filter kernel block */
MMatrix gaussianKernel(uint radius, int sigma);

/* Construct sobel filter kernel block */
MMatrix sobelKernel(size_t winSize);

/* Threshold filter */
MMatrix threshold(MMatrix* depthMat, double thres);

/* Gradient Compression */
void compressed(MMatrix* depthMat, double thres,  double alpha);

/* Solving Poisson equation to estimate 2D integrals*/
MMatrix IntgralSolver(MMatrix* V1, MMatrix* rho1, double accuracy);

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLE
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// Algorithm 1 : Adjust depth intensity + Gaussian Blur
///////////////////////////////////////////////////////////////////////////////

MMatrix gaussian(double intenSacle, MMatrix* depthMat, uint radius, int sigma)
{
	depthMat->mul(intenSacle);

	MMatrix kernel = gaussianKernel(radius, sigma);

	MMatrix retMat = filter(depthMat, kernel);

	return retMat;
}

///////////////////////////////////////////////////////////////////////////////
// Algorithm 2 : Bas-Relief -> Gradient compression
// Weyrich, Tim, et al. "Digital bas-relief from 3D scenes." 
// ACM Transactions on Graphics(TOG).Vol. 26. No. 3. ACM, 2007.
///////////////////////////////////////////////////////////////////////////////

MMatrix basRelief(MMatrix* depthMat, uint radius, double thres, double alpha)
{
	double accuracy = 0.00001; // Accuracy of integration approximation

	// Select Kernel for matrix differeniation

	// Sobel Kernel
	//MMatrix kernel = sobelKernel(2*radius+1);

	// Forward Difference Kernel (3-by-3)
	MMatrix fwdKer(3, 3, 0.0);
	fwdKer.setElement(1, 1, -1);
	fwdKer.setElement(1, 2, 1);

	// Backward Difference Kernel (3-by-3)
	MMatrix bkdKer(3, 3, 0.0);
	bkdKer.setElement(1, 0, -1);
	bkdKer.setElement(1, 1, 1);

	// Acquire map Forward Difference (Step I)
	MMatrix diffX(0, 0);
	MMatrix diffY(0, 0);
	MMatrix diffMag(0, 0);

	std::tie(diffX, diffY, diffMag) = matrixDiff(depthMat, fwdKer, true);

	//diffX = filter(depthMat, fwdKer, std::make_tuple(0, 0, 0, -1));
	////diffX.display();
	//diffY = filter(depthMat, ~fwdKer, std::make_tuple(0, -1, 0, 0));
	////diffY.display();
	//diffMag = diffX.times(diffX) + diffY.times(diffY);
	//diffMag.sqroot(); // $sqrt{ x^2 + y^2 }$

	// Gradient Compression (change only the gradient magnitude)  (Step II)
	compressed(&diffMag, thres, alpha);

	writeMatrix(&diffMag, "modifedMap.txt");

	// g' = s' $\times$ v'
	diffX *= diffMag; // gradient direction x times amplitude
	diffY *= diffMag; // gradient direction y times amplitude
	//diffX.display();
	//diffY.display();

	// Integration  (Step III)

	// Acquire map Backward Difference
	MMatrix divGx = filter(&diffX, bkdKer);
	MMatrix divGy = filter(&diffY, ~bkdKer);
	//divGx.display();
	//divGy.display();
	MMatrix divG = divGx + divGy;
	//divG.display();

	//MMatrix initMat = *depthMat;
	//std::cout << "Acquire Integration " << std::endl;
	//MMatrix iMat = IntgralSolver(&initMat, &divG, 0.01);
	//iMat.display();
	// Error (display all)
	//std::cout << "Error " << std::endl;
	//(iMat - (*depthMat)).display();

	MMatrix initMat = *depthMat;
	//initMat.setBlock(1.0); // Just a test
	//initMat.setBlock(0.0, std::make_tuple(1, -1, 1, -1)); // Just a test
	
	MMatrix retMat = IntgralSolver(&initMat, &divG, accuracy);

	// Calculate error
	double error = 0.0;
	for (int i = 0; i < depthMat->getRowsNum(); i++)
		for (int j = 0; j < depthMat->getColsNum(); j++)
			error += (depthMat->getElement(i, j) - retMat.getElement(i, j)) / depthMat->getElement(i, j)*100;
	std::cout << "Total error = " << error << " % " << std::endl
		<< "Average error = " << error / (depthMat->getRowsNum() * depthMat->getColsNum()) << " % " << std::endl;

	return retMat;
}

// Edge detection (matrix differentiation)
M3MatPtr matrixDiff(MMatrix* depthMat, MMatrix ker, bool isDirect)
{
	// gradient X direction
	MMatrix diffX = filter(depthMat, ker);

	// Display kernel
	std::cout << "Matrix differenitation with Kernel:" << std::endl;
	ker.display();

	// gradient Y direction
	MMatrix diffY = filter(depthMat, (~ker));

	// gradient magnitude
	MMatrix diffMag(depthMat->getRowsNum(), depthMat->getColsNum(), 0.0);

	diffMag = diffX.times(diffX) + diffY.times(diffY);
	diffMag.sqroot(); // $sqrt{ x^2 + y^2 }$

	if (isDirect == true) // Choose whether normalized x and y
	{
		diffX /= diffMag; // gradient direction x
		diffY /= diffMag; // gradient direction y
	}

	// diffMag = threshold(*diffX, 0.015); // Edge detection

	return std::make_tuple(diffX, diffY, diffMag); // Pack multiple return as tuple
}

///////////////////////////////////////////////////////////////////////////////
// ASSISTIVE FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

/* Import the original depth map image */
void readMatrix(MMatrix* mat, std::string filepath)
{
	std::ifstream inFile;
	std::string str;

	size_t height = mat->getRowsNum();
	size_t width = mat->getColsNum();

	inFile.open(filepath);
	for (uint i = 0; i < height; i++)
	{
		for (uint j = 0; j < width; j++)
		{
			getline(inFile, str, ' ');
			mat->setElement(i, j, stof(str));
		}
	}
	inFile.close();
}

/* Export the mapped image to a .txt file */
void writeMatrix(MMatrix* mat, std::string filename)
{
	size_t height = mat->getRowsNum();
	size_t width = mat->getColsNum();

	std::ofstream outFile;
	outFile.open(filename);
	for (uint i = 0; i < height; i++)
	{
		for (uint j = 0; j < width; j++)
		{
			outFile << mat->getElement(i, j) << " ";
		}
		outFile << std::endl;
	}
	outFile.close();
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

/* Apply filter to input data matrix */
MMatrix filter(MMatrix* mat, MMatrix ker, Range2D filtRange)
{
	int height = mat->getRowsNum();
	int width = mat->getColsNum();

	int rInit, rEnd, cInit, cEnd;

	// Optional argument: filter range
	if ((std::get<0>(filtRange) == 0) && (std::get<1>(filtRange) == 0) &&
		(std::get<2>(filtRange) == 0) && (std::get<3>(filtRange) == 0) )
	{
		std::cout << "Filtering entire matrix" << std::endl;
		rInit = 0;
		rEnd = height;
		cInit = 0;
		cEnd = width;
	}
	else
	{
		if (std::get<1>(filtRange) >= height)
		{
			std::cerr << "Warning: 'rEnd' out of matrix boundary" << std::endl;
			rEnd = height;
		}
		else if (std::get<1>(filtRange) <= 0)
		{
			rEnd = int(height) + std::get<1>(filtRange);
			if (rEnd <= 0) rEnd = height;
		}
		else
		{
			rEnd = std::get<1>(filtRange) +1;
		}

		if ( (std::get<0>(filtRange) >= rEnd) || (std::get<0>(filtRange) < 0) )
		{
			std::cerr << "Warning: illegal 'rInit' value" << std::endl;
			rInit = 0;
		}
		else
		{
			rInit = std::get<0>(filtRange);
		}

		if (std::get<3>(filtRange) >= width)
		{
			std::cerr << "Warning: 'cEnd' out of matrix boundary" << std::endl;
			cEnd = width;
		}
		else if (std::get<3>(filtRange) <= 0)
		{
			cEnd = int(width) + std::get<3>(filtRange);
			if (cEnd <= 0) cEnd = width;
		}
		else
		{
			cEnd = std::get<3>(filtRange) +1;
		}

		if ( (std::get<2>(filtRange) >= cEnd) || (std::get<2>(filtRange) < 0) )
		{
			std::cerr << "Warning: illegal 'cInit' value" << std::endl;
			cInit = 0;
		}
		else
		{
			cInit = std::get<2>(filtRange);
		}

		std::cout << "Filtering in Row " << rInit << " to " << rEnd-1
			<< " , Col " << cInit << " to " << cEnd-1 << std::endl;
	}

	size_t kerLen = ker.getRowsNum();

	int radius;
	int mod; // Modify the range of kernel for odd or even sites

	//double deftWeightSum = 0; // default sum of kernel weight (REMOVED)

	MMatrix mappedMat = *mat;

	/*for (uint p = 0; p < kerLen; p++)
	{
		for (uint q = 0; q < kerLen; q++)
		{
			deftWeightSum += (ker.getElement(p, q));
		}
	}*/ // (REMOVED)

	if (kerLen % 2 == 1) // If kernel side length is odd
	{
		radius = (kerLen - 1) / 2;
		mod = 0;
	}
	else // If kernel side length is even
	{
		radius = kerLen / 2 - 1;
		mod = 1;
	}

		// Apply filter
	for (int i = rInit; i < rEnd; i++)
	{
		for (int j = cInit; j < cEnd; j++)
		{
			int mInit = -radius;
			int mEnd = radius + mod;
			int nInit = -radius;
			int nEnd = radius + mod;
			//double weightSum = 0; // current sum of kernel weight  (REMOVED)

			// if filter block exceed image boundary then truncate filter block
			if (i < radius){ mInit = -i; }
			if (i >= (height - radius - mod)){ mEnd = height - 1 - i; }
			if (j < radius){ nInit = -j; }
			if (j >= (width - radius - mod)){ nEnd = width - 1 - j; }

			double filtSum = 0;

			for (int m = mInit; m <= mEnd; m++)
			{
				for (int n = nInit; n <= nEnd; n++)
				{
					// convolution
					filtSum += mat->getElement(i + m, j + n) * ker.getElement(m + radius, n + radius);
					//weightSum += (ker.getElement(m + radius, n + radius));
				}
			}

			//if ((weightSum < deftWeightSum) && (weightSum != 0)) //  (REMOVED)
			//{
			//	// In case of kernel block being truncated
			//	mappedMat.setElement(i, j, (filtSum * deftWeightSum / weightSum));
			//}
			//else // (REMOVED on 01/15/2016 due to \pm issues)
			//{
				mappedMat.setElement(i, j, filtSum);
			//}
		}
	}	
	return mappedMat;
}

/* Construct gaussian filter kernel block */
MMatrix gaussianKernel(uint radius, int sigma)
{
	size_t kerLen = 2 * radius + 1; // kerLen * kerLen square matrix
	double sigma2 = 2 * sigma * sigma; // gaussian parameter sigma

	int mu = (kerLen - 1) / 2; // zero mean 

	double kerSum = 0;

	MMatrix ker(kerLen, kerLen, 0.0);

	//ker = new double*[kerLen];

	for (int i = 0; i < kerLen; i++)
	{
		//ker[i] = new double[kerLen];
		for (int j = 0; j < kerLen; j++)
		{
			// The formula: kernel weight = e^{ -( ((i- \mu)^2+(j-\mu)^2) )/( 2* \sigma^2 )}
			ker.setElement(i, j, (exp(-((i - mu)*(i - mu) + (j - mu)*(j - mu)) / sigma2)) );

			kerSum += ker.getElement(i, j);
		}
	}

	ker.div(kerSum);

	for (uint i = 0; i < kerLen; i++)
	{
		for (uint j = 0; j < kerLen; j++)
		{
			ker.setElement(i, j,  (floor(ker.getElement(i, j) * 1000000.0) / 1000000.0) );
			//cout << ker[i][j] << " "; // for test only
		}
		std::cout << std::endl;
	}

	//ker.display(); // For test only
	return ker;
}

/* Construct sobel filter kernel block (x direction) */
MMatrix sobelKernel(size_t winSize)
{
	// Construct a smooth operater (row vector)
	MVector smoothOperator = pascalTriangle(winSize, 1.0, 1.0);

	// Construct a difference operater (row vector)
	MVector diffOperator = pascalTriangle(winSize, 1.0, -1.0);

	// Sobel kernal matrix = $ (smooth operater)^T \cdot difference operater $
	MMatrix ker = (~smoothOperator) * diffOperator;

	return ker;
}

/* Threshold filter */
MMatrix threshold(MMatrix* depthMat, double thres)
{
	MMatrix retMat = *depthMat;

	double offset = depthMat->min();

	double range = depthMat->max() - offset;

	// Normalization of the matrix
	retMat.sub(offset); // minus minimum value
	retMat.div(range); // divided by range

	retMat = retMat.isGreator(thres);

	return retMat;
}

/* Gradient Compression */ 
// C(x) = \frac{1}{\alpha}\times\log(1+\alpha\times x)
void compressed(MMatrix* depthMat, double thres, double alpha)
{
	for (uint i = 0; i < depthMat->getRowsNum(); i++)
	{
		for (uint j = 0; j < depthMat->getColsNum(); j++)
		{
			if (depthMat->getElement(i, j) < thres) // less than threshold
			{
				depthMat->setElement(i, j, (log(1 + alpha * depthMat->getElement(i, j)) / alpha) );

				//cout << depthMat->getElement(i, j) << endl; // For test only 01/08/2016
			}
			else
			{
				depthMat->setElement(i, j, 0.0);
			}
		}
	}
}

/* Solving Poisson equation: \triangledown^2 V = \rho */
MMatrix IntgralSolver(MMatrix* V1, MMatrix* rho1, double accuracy)
{
	size_t height = V1->getRowsNum();
	size_t width = V1->getColsNum();

	MMatrix V1_new = *V1; // Updated matrix

	clock_t t0 = clock(); // Initial time of the solver

	int steps = 0; //  count iteration steps
	double omega = 2 / (1 + M_PI / sqrt(height*width)); // For SOR method only
	bool continueItr = true; // whether the iteration continues

	std::cout << "Solving Equation ..." << std::endl;

	while (continueItr)
	{
		///* Gauss-Seidel method */
		//*V1_new = *V1;
		//for (int i = 1; i <= height - 2; i++)
		//{
		//	for (int j = 1; j <= width - 2; j++)
		//	{
		//		V1_new->setElement(i, j, 0.25*(V1_new->getElement(i - 1, j) 
		//			+ V1_new->getElement(i + 1, j)
		//			+ V1_new->getElement(i, j - 1) 
		//			+ V1_new->getElement(i, j + 1) - rho1->getElement(i, j)));
		//	}
		//}

		/* Successive Over Relaxation (SOR) method */
		for (uint i = 1; i <= height - 2; i++)
		{
			for (uint j = 1; j <= width - 2; j++)
			{
				if ((i + j) % 2 == 0) // Update even sites
				{
					V1_new.setElement(i, j, (1 - omega) * V1->getElement(i, j)
						+ omega * 0.25 * (V1->getElement(i - 1, j) + V1->getElement(i + 1, j)
						+ V1->getElement(i, j - 1) + V1->getElement(i, j + 1)
						- rho1->getElement(i, j)));
					//std::cout << (1 - omega) << " * " << V1->getElement(i, j) << " + " 
					//	<< omega * 0.25
					//	<< "* ( "
					//	<< V1->getElement(i - 1, j) << " + "
					//	<< V1->getElement(i + 1, j) << " + "
					//	<< V1->getElement(i, j - 1) << " + "
					//	<< V1->getElement(i, j + 1) << " - "
					//	<< rho1->getElement(i, j) << " ) = "
					//	<< (1 - omega) * V1->getElement(i, j)
					//	+ omega * 0.25 * (V1->getElement(i - 1, j) + V1->getElement(i + 1, j)
					//	+ V1->getElement(i, j - 1) + V1->getElement(i, j + 1)
					//	- rho1->getElement(i, j))
					//	<< std::endl;
					//std::cout << " " << V1->getElement(i, j) << std::endl;
				}
			}
		}

		for (uint i = 1; i <= height - 2; i++)
		{
			for (uint j = 1; j <= width - 2; j++)
			{
				if ((i + j) % 2 != 0) // Update odd sites
					V1_new.setElement(i, j, (1 - omega) * V1->getElement(i, j)
					+ omega * 0.25 * (V1_new.getElement(i - 1, j) + V1_new.getElement(i + 1, j)
					+ V1_new.getElement(i, j - 1) + V1_new.getElement(i, j + 1)
					- rho1->getElement(i, j)));
			}
		}

		double error = 0;
		int n = 0;

		// Compute error
		for (int i = 1; i <= height - 2; i++)
		{
			for (int j = 1; j <= width - 2; j++)
			{
				double oldVal = V1->getElement(i, j);
				double newVal = V1_new.getElement(i, j);
				if (newVal != 0)
					if (newVal != oldVal)
					{
						error += abs(1 - oldVal / newVal);
						n++;
					}
			}
		}
		//std::cout << error << " , n = " << n << std::endl;

		if (n != 0) error /= n;

		if (error < accuracy)
		{
			continueItr = false;
		}
		else
		{
			*V1 = V1_new;
		}

		steps++;
	}

	std::cout << "Number of steps = " << steps << std::endl;
	std::cout << "CPU time = " << double(clock() - t0) / CLOCKS_PER_SEC << " sec" << std::endl;

	return V1_new;
}


///////////////////////////////////////////////////////////////////////////////
// TestOnly FUNCTIONS
///////////////////////////////////////////////////////////////////////////////
void test()
{

}
//void test()
//{
//	// Test Poisson equation solver on 01/11/2016
//	int L = 20;
//	MMatrix* V = new MMatrix(L , L , 0.0);
//	for (uint i = 0; i < L; i++)
//		for (uint j = 0; j < L; j++)
//			V->setElement(i, j, rand()%10+1);
//	V->display();
//
//	//MMatrix kernel = sobelKernel(2 * 1 + 1);
//
//	// Forward Difference Kernel (3-by-3)
//	MMatrix fwdKer(3, 3, 0.0);
//	fwdKer.setElement(1, 1, 1);
//	fwdKer.setElement(1, 2, -1);
//
//	// Backward Difference Kernel (3-by-3)
//	MMatrix bkdKer(3, 3, 0.0);
//	bkdKer.setElement(1, 0, 1);
//	bkdKer.setElement(1, 1, -1);
//
//	// // Acquire map Forward Difference 
//	//MMatrix diffX = filter(V, fwdKer, std::make_tuple(0,0,0,-1));
//	////diffX.display();
//	//MMatrix diffY = filter(V, ~fwdKer, std::make_tuple(0, -1, 0, 0));
//	////diffY.display();
//
//	MMatrix diffX(L, L, 0.0);
//	MMatrix diffY(L, L, 0.0);
//	MMatrix diffMag(L, L, 0.0);
//
//	std::tie(diffX, diffY, diffMag) = matrixDiff(V, fwdKer, true);
//
//	// Gradient Compression (change only the gradient magnitude)  (Step II)
//	compressed(&diffMag, 100, 5.0);
//
//	// g' = s' $\times$ v'
//	diffX *= diffMag;
//	diffY *= diffMag;
//
//	// Acquire map Backward Difference 
//	std::cout << "Acquire map Backward Difference " << std::endl;
//	MMatrix divGx = filter(&diffX, bkdKer);
//	//divGx.display();
//	MMatrix divGy = filter(&diffY, ~bkdKer);
//	//divGy.display();
//	MMatrix divG = divGx + divGy;
//	//divG.mul( 0.9 );
//	divG.display();
//
//	// Acquire Integration
//	MMatrix initMat = *V;
//	std::cout << "Acquire Integration " << std::endl;
//	MMatrix iMat = IntgralSolver(&initMat, &divG, 0.01);
//	iMat.display();
//
//	// Error
//	std::cout << "Error " << std::endl;
//	(iMat - (*V)).display();
//}
