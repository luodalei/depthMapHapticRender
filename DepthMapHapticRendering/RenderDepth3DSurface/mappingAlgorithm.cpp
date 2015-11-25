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
void readMatrix(double** mat, int width, int height, string filepath);
void writeMatrix(double** mat, int width, int height, string filename);
double** filter(double** mat, double** ker);
double** gaussianKernel(int radius, int sigma);

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLE
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// Algorithm 1 : Adjust depth intensity + Gaussian Blur
///////////////////////////////////////////////////////////////////////////////
double** gaussian(double intenSacle, double** depthMat, int radius, int sigma)
{
	double** retMat;

	for (int i = 0; i < IMAGE_HEIGHT; i++)
	{
		for (int j = 0; j < IMAGE_WIDTH; j++)
		{
			depthMat[i][j] = depthMat[i][j] * intenSacle;
		}
	}

	double** kernel = gaussianKernel(radius, sigma);

	retMat = filter(depthMat, kernel);

	return retMat;
}

///////////////////////////////////////////////////////////////////////////////
// ASSISTIVE FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

/* Import the original depth map image */
void readMatrix(double** mat, int width, int height, string filepath)
{
	ifstream inFile;
	string str;

	inFile.open(filepath);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			getline(inFile, str, ' ');
			mat[i][j] = stof(str);
		}
	}
	inFile.close();
}

/* Export the mapped image to a .txt file */
void writeMatrix(double** mat, int width, int height, string filename)
{
	ofstream outFile;
	outFile.open(filename);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			outFile << mat[i][j] << " ";
		}
		outFile << endl;
	}
	outFile.close();
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

/* Apply filter to input data matrix */
double** filter(double** mat, double** ker)
{
	int radius = sizeof(ker);
	int kerLen = 2 * radius + 1;
	double deftWeightSum = 0; // default sum of kernel weight

	double** mappedMat = 0;
	mappedMat = new double*[IMAGE_HEIGHT];

	// initialize an empty mapped matrix
	for (int i = 0; i < IMAGE_HEIGHT; i++) 
	{
		mappedMat[i] = new double[IMAGE_WIDTH];
		for (int j = 0; j < IMAGE_WIDTH; j++)
		{
			mappedMat[i][j] = 0;
		}
	}

	for (int p = 0; p < kerLen; p++)
	{
		for (int q = 0; q < kerLen; q++)
		{
			deftWeightSum += ker[p][q];
		}
	}

	// Apply filter
	for (int i = 0; i < IMAGE_HEIGHT; i++)
	{
		for (int j = 0; j < IMAGE_WIDTH; j++)
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
					// Average filter
					filtSum += mat[i + m][j + n] * ker[m + radius][n + radius];
					weightSum += ker[m + radius][n + radius];
				}
			}
	
			if (weightSum < deftWeightSum) 
			{
				// In case of kernel block being truncated
				mappedMat[i][j] = filtSum * deftWeightSum / weightSum;
			}
			else 
			{
				mappedMat[i][j] = filtSum;
			}
		}
	}

	return mappedMat;
}

double** gaussianKernel(int radius, int sigma)
{
	int kerLen = 2 * radius + 1;
	int sigma2 = 2 * sigma;

	int mu = (kerLen - 1) / 2;

	double kerSum = 0;

	double** ker = 0;
	ker = new double*[kerLen];

	for (int i = 0; i < kerLen; i++)
	{
		ker[i] = new double[kerLen];
		for (int j = 0; j < kerLen; j++)
		{
			ker[i][j] = exp(-((i - mu)*(i - mu) + (j - mu)*(j - mu)) / sigma2);
			kerSum += ker[i][j];
		}
	}
	for (int i = 0; i < kerLen; i++)
	{
		for (int j = 0; j < kerLen; j++)
		{
			ker[i][j] /= kerSum;
			ker[i][j] = floor(ker[i][j] * 1000000.0) / 1000000.0;
			//cout << ker[i][j] << " "; // for test only
		}
		cout << endl;
	}
	return ker;
}