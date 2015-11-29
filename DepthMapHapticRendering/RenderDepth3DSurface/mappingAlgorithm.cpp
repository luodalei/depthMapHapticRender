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

#include <iostream> 
#include <fstream> 
#include <string>

//------------------------------------------------------------------------------
using namespace std;

///////////////////////////////////////////////////////////////////////////////
// FUNCTION DECLARATION 
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
	// Acquire map gradient
	MMatrix** matrixGradient = matrixDiff(depthMat, radius);

	MMatrix* diffX = matrixGradient[0];
	MMatrix* diffY = matrixGradient[1];
	MMatrix* diffMag = matrixGradient[2];

	// Gradient Compression (change only the gradient magnitude)
	compressed(diffMag, thres, alpha);

	// g' = s' $\times$ v'
	(*diffX) *= (*diffMag); 
	(*diffY) *= (*diffMag);

	// Integration

	MMatrix* retMat;

	return retMat;
}

// Edge detection (matrix differentiation)
MMatrix** matrixDiff(MMatrix depthMat, uint radius)
{
	MMatrix* retMat[3];

	MMatrix kernel = sobelKernel(2*radius+1);

	kernel.display();

	// diffX
	retMat[0] = filter(depthMat, kernel);

	// diffY
	retMat[1] = filter(depthMat, (~kernel));

	MMatrix* diffMag = new MMatrix(depthMat.getRowsNum(), depthMat.getColsNum(), 0.0);

	(*diffMag) = ( retMat[0]->times(*retMat[0]) ) + ( retMat[1]->times(*retMat[1]) );
	diffMag->sqroot(); // $sqrt{ x^2 + y^2 }$

	(*retMat[0]) /= (*diffMag); // gradient direction x
	(*retMat[1]) /= (*diffMag); // gradient direction y
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
	int radius = (kerLen - 1) / 2;
	
	double deftWeightSum = 0; // default sum of kernel weight

	MMatrix* mappedMat = new MMatrix(height, width, 0.0);

	for (uint p = 0; p < kerLen; p++)
	{
		for (uint q = 0; q < kerLen; q++)
		{
			deftWeightSum += ker.getElement(p,q);
		}
	}

	// Apply filter
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int mInit = -radius;
			int mEnd = radius;
			int nInit = -radius;
			int nEnd = radius;
			double weightSum = 0; // current sum of kernel weight

			// if filter block exceed image boundary then truncate filter block
			if (i < radius){ mInit = -i; }
			if (i >= (height - radius)){ mEnd = height - i; }
			if (j < radius){ nInit = -j; }
			if (j >= (width - radius)){ nEnd = width - j; }

			double filtSum = 0;

			for (int m = mInit; m < mEnd; m++)
			{
				for (int n = nInit; n < nEnd; n++)
				{
					// convolution
					filtSum += mat.getElement(i + m, j + n) * ker.getElement(m + radius, n + radius);
					weightSum += ker.getElement(m + radius, n + radius);
				}
			}
	
			if (weightSum < deftWeightSum) 
			{
				// In case of kernel block being truncated
				mappedMat->setElement(i, j, (filtSum * deftWeightSum / weightSum) );
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
			}
			else
			{
				depthMat->setElement(i, j, 0.0);
			}
		}
	}
}


///////////////////////////////////////////////////////////////////////////////
// TestOnly FUNCTIONS
///////////////////////////////////////////////////////////////////////////////
void test()
{
	//sobelKernel(5);
}