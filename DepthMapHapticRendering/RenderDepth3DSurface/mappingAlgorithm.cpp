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

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLE
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// Algorithm 1 : Adjust depth intensity + Gaussian Blur
///////////////////////////////////////////////////////////////////////////////

MMatrix* gaussian(double intenSacle, MMatrix depthMat, uint radius, int sigma)
{
	size_t height = depthMat.getRowsNum();
	size_t width = depthMat.getColsNum();

	for (uint i = 0; i < height; i++)
	{
		for (uint j = 0; j < width; j++)
		{
			depthMat.setElement(i, j, depthMat.getElement(i, j) * intenSacle);
		}
	}

	MMatrix kernel = gaussianKernel(radius, sigma);

	MMatrix* retMat = filter(depthMat, kernel);

	return retMat;
}

///////////////////////////////////////////////////////////////////////////////
// Algorithm 2 : Bas-Relief -> Gradient compression
// Weyrich, Tim, et al. "Digital bas-relief from 3D scenes." 
// ACM Transactions on Graphics(TOG).Vol. 26. No. 3. ACM, 2007.
///////////////////////////////////////////////////////////////////////////////

MMatrix gradient(double intenSacle, MMatrix depthMat, int radius)
{
	MMatrix retMat(IMAGE_HEIGHT, IMAGE_WIDTH, 0.0);

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
	size_t kerLen = ker.getRowsNum();
	int radius = (kerLen - 1) / 2;
	
	double deftWeightSum = 0; // default sum of kernel weight

	MMatrix* mappedMat = new MMatrix(IMAGE_HEIGHT, IMAGE_WIDTH, 0.0);

	for (uint p = 0; p < kerLen; p++)
	{
		for (uint q = 0; q < kerLen; q++)
		{
			deftWeightSum += ker.getElement(p,q);
		}
	}

	// Apply filter
	for (uint i = 0; i < IMAGE_HEIGHT; i++)
	{
		for (uint j = 0; j < IMAGE_WIDTH; j++)
		{
			int mInit = -radius;
			int mEnd = radius;
			int nInit = -radius;
			int nEnd = radius;
			double weightSum = 0; // current sum of kernel weight

			// if filter block exceed image boundary then truncate filter block
			if (i < radius){ mInit = -i; }
			if (i >= (IMAGE_HEIGHT - radius)){ mEnd = IMAGE_HEIGHT - i; }
			if (j < radius){ nInit = -j; }
			if (j >= (IMAGE_WIDTH - radius)){ nEnd = IMAGE_WIDTH - j; }		

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
			// kernel weight = e^{ -( ((i- \mu)^2+(j-\mu)^2) )/( 2* \sigma^2 )}
			ker.setElement(i, j, (exp(-((i - mu)*(i - mu) + (j - mu)*(j - mu)) / sigma2)) );
			kerSum += ker.getElement(i, j);
		}
	}
	for (uint i = 0; i < kerLen; i++)
	{
		for (uint j = 0; j < kerLen; j++)
		{
			ker.setElement(i, j, (ker.getElement(i, j) / kerSum)); // Update needed in the future !!!
			ker.setElement(i, j,  (floor(ker.getElement(i, j) * 1000000.0) / 1000000.0) );
			//cout << ker[i][j] << " "; // for test only
		}
		cout << endl;
	}
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


///////////////////////////////////////////////////////////////////////////////
// TestOnly FUNCTIONS
///////////////////////////////////////////////////////////////////////////////
void test()
{
	//sobelKernel(5);
}