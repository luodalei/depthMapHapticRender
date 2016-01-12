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
#include <time.h> // Evaluate algorithm by time

//------------------------------------------------------------------------------
using namespace std;

///////////////////////////////////////////////////////////////////////////////
// FUNCTION (LOCAL USE ONLY) DECLARATION 
///////////////////////////////////////////////////////////////////////////////

MMatrix* filter(MMatrix mat, MMatrix ker);

/* Construct gaussian filter kernel block */
MMatrix gaussianKernel(uint radius, int sigma);

/* Construct sobel filter kernel block */
MMatrix sobelKernel(size_t winSize);

/* Threshold filter */
MMatrix* threshold(MMatrix depthMat, double thres);

/* Gradient Compression */
void compressed(MMatrix* depthMat, double thres,  double alpha);

/* Solving Poisson equation to estimate 2D integrals*/
MMatrix* IntgralSolver(MMatrix* V1, MMatrix* rho1, double accuracy);

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLE
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// Algorithm 1 : Adjust depth intensity + Gaussian Blur
///////////////////////////////////////////////////////////////////////////////

MMatrix* gaussian(double intenSacle, MMatrix depthMat, uint radius, int sigma)
{
	depthMat.mul(intenSacle);

	MMatrix kernel = gaussianKernel(radius, sigma);

	MMatrix* retMat = filter(depthMat, kernel);

	return retMat;
}

///////////////////////////////////////////////////////////////////////////////
// Algorithm 2 : Bas-Relief -> Gradient compression
// Weyrich, Tim, et al. "Digital bas-relief from 3D scenes." 
// ACM Transactions on Graphics(TOG).Vol. 26. No. 3. ACM, 2007.
///////////////////////////////////////////////////////////////////////////////

MMatrix* basRelief(MMatrix depthMat, uint radius, double thres, double alpha)
{
	double accuracy = 0.001; // Accuracy of integration approximation

	// Select Kernel for matrix differeniation

	// Sobel Kernel
	//MMatrix kernel = sobelKernel(2*radius+1);

	// Forward Difference Kernel (3-by-3)
	MMatrix fwdKer(3, 3, 0.0);
	fwdKer.setElement(1, 1, 1);
	fwdKer.setElement(1, 2, -1);

	// Backward Difference Kernel (3-by-3)
	MMatrix bkdKer(3, 3, 0.0);
	bkdKer.setElement(1, 0, -1);
	bkdKer.setElement(1, 1, 1);

	// Acquire map Forward Difference (Step I)
	MMatrix** matrixGradient = matrixDiff(depthMat, fwdKer, false);

	MMatrix* diffX = matrixGradient[0];
	MMatrix* diffY = matrixGradient[1];
	MMatrix* diffMag = matrixGradient[2];

	// Gradient Compression (change only the gradient magnitude)  (Step II)
	compressed(diffMag, thres, alpha);

	// g' = s' $\times$ v'
	(*diffX) *= (*diffMag); 
	(*diffY) *= (*diffMag);

	// Integration  (Step III)

	// Acquire map Backward Difference
	MMatrix** divG = matrixDiff(*diffMag, bkdKer, false);

	MMatrix* V = new MMatrix(depthMat.getRowsNum(), depthMat.getColsNum(), 0.0);
	MMatrix* rho = divG[2];

	MMatrix* retMat = IntgralSolver(V, rho, accuracy);

	return retMat;
}

// Edge detection (matrix differentiation)
MMatrix** matrixDiff(MMatrix depthMat, MMatrix ker, bool isDirect)
{
	cout << "Matrix differenitation with Kernel:" << endl;

	MMatrix* retMat[3];

	// diffX
	retMat[0] = filter(depthMat, ker);
	ker.display();

	// diffY
	retMat[1] = filter(depthMat, (~ker));

	MMatrix* diffMag = new MMatrix(depthMat.getRowsNum(), depthMat.getColsNum(), 0.0);

	(*diffMag) = (retMat[0]->times(*retMat[0])) + (retMat[1]->times(*retMat[1]));
	diffMag->sqroot(); // $sqrt{ x^2 + y^2 }$

	if (isDirect == true) // Choose whether normalized x and y
	{
		(*retMat[0]) /= (*diffMag); // gradient direction x
		(*retMat[1]) /= (*diffMag); // gradient direction y
	}

	//retMat[0] -> display();
	//retMat[0] -> display();

	retMat[2] = diffMag; // gradient magnitude

	//MMatrix* retMat = threshold(*diffX, 0.015); // Edge detection

	return retMat;
}

///////////////////////////////////////////////////////////////////////////////
// ASSISTIVE FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

/* Import the original depth map image */
void readMatrix(MMatrix* mat, string filepath)
{
	ifstream inFile;
	string str;

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
void writeMatrix(MMatrix* mat, string filename)
{
	size_t height = mat->getRowsNum();
	size_t width = mat->getColsNum();

	ofstream outFile;
	outFile.open(filename);
	for (uint i = 0; i < height; i++)
	{
		for (uint j = 0; j < width; j++)
		{
			outFile << mat->getElement(i, j) << " ";
		}
		outFile << endl;
	}
	outFile.close();
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

/* Apply filter to input data matrix */
MMatrix* filter(MMatrix mat, MMatrix ker)
{
	size_t height = mat.getRowsNum();
	size_t width = mat.getColsNum();

	size_t kerLen = ker.getRowsNum();

	int radius;
	int mod; // Modify the range of kernel for odd or even sites
	double deftWeightSum = 0; // default sum of kernel weight

	MMatrix* mappedMat = new MMatrix(height, width, 0.0);

	for (uint p = 0; p < kerLen; p++)
	{
		for (uint q = 0; q < kerLen; q++)
		{
			deftWeightSum += ker.getElement(p, q);
		}
	}

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
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int mInit = -radius;
			int mEnd = radius + mod;
			int nInit = -radius;
			int nEnd = radius + mod;
			double weightSum = 0; // current sum of kernel weight

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
					filtSum += mat.getElement(i + m, j + n) * ker.getElement(m + radius, n + radius);
					weightSum += ker.getElement(m + radius, n + radius);
				}
			}

			if ((weightSum < deftWeightSum) && (deftWeightSum != 0) && (weightSum != 0))
			{
				// In case of kernel block being truncated
				mappedMat->setElement(i, j, (filtSum * deftWeightSum / weightSum));
			}
			else
			{
				mappedMat->setElement(i, j, filtSum);
			}
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
		cout << endl;
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
MMatrix* threshold(MMatrix depthMat, double thres)
{
	MMatrix* retMat = new MMatrix(depthMat.getRowsNum(), depthMat.getColsNum(), 0.0);

	double offset = depthMat.min();

	double range = depthMat.max() - offset;

	// Normalization of the matrix
	depthMat.sub(offset); // minus minimum value
	depthMat.div(range); // divided by range

	*retMat = depthMat.isGreator(thres);

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
MMatrix* IntgralSolver(MMatrix* V1, MMatrix* rho1, double accuracy)
{
	size_t height = V1->getRowsNum();
	size_t width = V1->getColsNum();

	MMatrix* V1_new = new MMatrix(height, width, 0.0); // Updated matrix

	clock_t t0 = clock(); // Initial time of the solver

	int steps = 0; //  count iteration steps
	double omega = 2 / (1 + M_PI / sqrt(height*width)); // For SOR method only
	bool continueItr = true; // whether the iteration continues

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
					V1_new->setElement(i, j, (1 - omega) * V1->getElement(i, j)
					+ omega * 0.25 * (V1->getElement(i - 1, j) + V1->getElement(i + 1, j)
					+ V1->getElement(i, j - 1) + V1->getElement(i, j + 1)
					- rho1->getElement(i, j)));
			}
		}
		for (uint i = 1; i <= height - 2; i++)
		{
			for (uint j = 1; j <= width - 2; j++)
			{
				if ((i + j) % 2 != 0) // Update odd sites
					V1_new->setElement(i, j, (1 - omega) * V1->getElement(i, j)
					+ omega * 0.25 * (V1_new->getElement(i - 1, j) + V1_new->getElement(i + 1, j)
					+ V1_new->getElement(i, j - 1) + V1_new->getElement(i, j + 1)
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
				double newVal = V1_new->getElement(i, j);
				if (newVal != 0)
					if (newVal != oldVal)
					{
						error += abs(1 - oldVal / newVal);
						n++;
					}
			}
		}

		if (n != 0) error /= n;

		if (error < accuracy)
		{
			continueItr = false;
		}
		else
		{
			MMatrix* tempt = V1;
			V1 = V1_new;
			V1_new = tempt;
		}

		steps++;
	}

	cout << "Number of steps = " << steps << endl;
	cout << "CPU time = " << double(clock() - t0) / CLOCKS_PER_SEC << " sec" << endl;

	return V1_new;
}


///////////////////////////////////////////////////////////////////////////////
// TestOnly FUNCTIONS
///////////////////////////////////////////////////////////////////////////////
void test()
{
	// Test Poisson equation solver on 01/11/2016
	int L = 10;
	MMatrix* V = new MMatrix(L , L , 0.0);
	for (uint i = 0; i < L; i++)
		for (uint j = 0; j < L; j++)
			V->setElement(i, j, i*L + j);
	//V->display();

	MMatrix kernel = sobelKernel(2 * 1 + 1);

	// Forward Difference Kernel (3-by-3)
	MMatrix fwdKer(3, 3, 0.0);
	fwdKer.setElement(1, 1, 1);
	fwdKer.setElement(1, 2, -1);

	// Backward Difference Kernel (3-by-3)
	MMatrix bkdKer(3, 3, 0.0);
	bkdKer.setElement(1, 0, -1);
	bkdKer.setElement(1, 1, 1);

	// Acquire map Forward Difference 
	MMatrix** divH = matrixDiff(*V, fwdKer, false);

	// Acquire map Backward Difference 
	MMatrix** divG = matrixDiff(*V, bkdKer, false);
}
