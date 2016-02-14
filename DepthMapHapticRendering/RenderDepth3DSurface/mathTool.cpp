//==============================================================================
/*
\author    Yitian Shao
\created 11/26/2015

Mathmatical Tool 
- Matrix Class
- Vector Class
- Generate pascal triangle
*/
//==============================================================================

//------------------------------------------------------------------------------
#include "mathTool.h"
//------------------------------------------------------------------------------

#include <iostream>
#include <vector>

//------------------------------------------------------------------------------
//// Define dype
//typedef unsigned int uint; // unsigned integer
//typedef std::vector<double> dbvector; // vector of double type
//typedef std::vector<double>::iterator dbiterator; // iterator of double type;
//typedef std::vector< std::vector< double > > dbmatrix; // matrix of double type	
//typedef std::tuple<uint, int, uint, int> Range2D; // Store range of rows and columns of a matrix
//typedef std::tuple<MMatrix, MMatrix, MMatrix> M3MatPtr; // 3 MMatrix tuple

///////////////////////////////////////////////////////////////////////////////
// Matrix class
///////////////////////////////////////////////////////////////////////////////

//class MMatrix // Declared in "mathTool.h"
//{
//	private:
//		// matrix with size of rowsNum $\times$ colsNum
//		size_t rowsNum; // number of rows
//		size_t colsNum; // number of columns
//
//	protected:
//		dbmatrix mMat; // the matrix
//
//	public:
//		// Constructor : method 0 -initialize only the row vector
//		//MMatrix(size_t m); // Not functioning, needed to be updated in future 
//
//		// Constructor : method 1 -initialize a matrix with m $\times$ n capacity
//		MMatrix(size_t m, size_t n);
//
//		// Constructor : method 2 -initialize a matrix with a value
//		MMatrix(size_t m, size_t n, double initVal); 
//
//		// Constructor : method 3 -convert an existing 2D array to matrix
//		//MMatrix(double** mat);  needed to be updated in future 
//
//		// Destructor
//		~MMatrix();
//
//		// Get matrix size (number of rows or columns)
//		size_t getRowsNum() const;
//		size_t getColsNum() const;
//
//		// Set matrix size (number of rows or columns)
//		void setRowsNum(size_t m);
//		void setColsNum(size_t n);
//
//		// set element at i, j of a value
//		void setElement(uint i, uint j, double val);
//
//		// set a block from rInit to rEnd and cInit to cEnd with a value
//		void setBlock(double val, Range2D blockRange);
//
//		// get value of element at i, j
//		double getElement(uint i, uint j);
//
//		//------------------------------------------------------------------ =
//		// Matrix operation: assignment (=) 
//		// Matrix dimension may subject to change
//		MMatrix& operator= (const MMatrix& assigned);
//
//		// Matrix operation: equal to (==)
//		bool operator== (const MMatrix& compared);
//
//		//------------------------------------------------------------------ >
//		// Matrix operation: element-wise greater than (>)
//		MMatrix operator>(const MMatrix& compared);
//
//		// Matrix operation: greater than threshold (>)
//		MMatrix isGreator(double thres);
//
//		//------------------------------------------------------------------ +
//		// Matrix operation: add (+=)
//		MMatrix& operator+= (const MMatrix& added);
//
//		// Matrix operation: add (+)
//		MMatrix operator+(const MMatrix& added);
//
//		// Matrix operation: added by single number
//		void add(double added);
//
//		//------------------------------------------------------------------ -
//		// Matrix operation: subtract (-=)
//		MMatrix& operator-= (const MMatrix& subtracted);
//
//		// Matrix operation: subtract (-)
//		MMatrix operator-(const MMatrix& subtracted);
//
//		// Matrix operation: subtracted by single number
//		void sub(double subtracted);
//
//		//------------------------------------------------------------------ *
//		// Matrix operation: element-wise multiplication (.*=)
//		MMatrix& operator*=(const MMatrix& multiplied);
//
//		// Matrix operation: element-wise multiplication (.*)
//		MMatrix times(const MMatrix& multiplied);
//
//		// Matrix operation: multiplied by single number
//		void mul(double multiplied);
//
//		// Matrix operation: inner product (*) 
//		MMatrix operator* (const MMatrix& multiplied);
//
//		//------------------------------------------------------------------ /
//		// Matrix operation: element-wise divide (./)
//		MMatrix& operator/=(const MMatrix& divided); // divided by a matrix
//
//		// Matrix operation: element-wise divide (/)
//		MMatrix operator/(const MMatrix& divided);
//
//		// Matrix operation: divided by single number
//		void div(double divided); 
//
//		//------------------------------------------------------------------ e
//		// Matrix operation: logarithm
//		void loga(double base);
//
//		//------------------------------------------------------------------ r
//		// Matrix operation: square root
//		void sqroot(void);
//
//		//------------------------------------------------------------------ T
//		// Matrix operation: transform (')
//		MMatrix operator~(void);
//
//		//------------------------------------------------------------------ :
//		// Matrix operation: truncation
//		MMatrix truncate(int row0, int row1, int col0, int col1);
//
//		//------------------------------------------------------------------
//		// Matrix operation: maximum value of whole matrix
//		double max(void);
//
//		// Matrix operation: minimum value of whole matrix
//		double min(void);
//
//		//  Display matrix in console (unsuitable for large matrix)
//		void display(void);
//};

// Constructor : method 0 // Not functioning, needed to be updated in future 
//MMatrix::MMatrix(size_t m)
//{
//	this->rowsNum = m;
//	this->mMat.reserve(this->rowsNum);
//}

// Constructor : method 1
MMatrix::MMatrix(size_t m, size_t n)
{
	this->rowsNum = m;
	this->colsNum = n;

	this->mMat.reserve(this->rowsNum);

	for (unsigned int i = 0; i < this->rowsNum; i++)
	{
		dbvector oneRow(colsNum); // initial value = 0.0
		this->mMat.push_back(oneRow);
	}
}

// Constructor : method 2
MMatrix::MMatrix(size_t m, size_t n, double initVal)
{
	this->rowsNum = m;
	this->colsNum = n;

	this->mMat.reserve(this->rowsNum);

	for (uint i = 0; i < this->rowsNum; i++)
	{
		dbvector oneRow(this->colsNum, initVal);
		this->mMat.push_back(oneRow);
	}
}

// Constructor : method 3 // needed to be updated in future 
//MMatrix::MMatrix(double** mat)
//{
//	this->rowsNum = sizeof(mat);
//	this->colsNum = sizeof(mat[0]);
//
//	this->mMat.reserve(this->rowsNum);
//
//	for (uint i = 0; i < this->rowsNum; i++)
//	{
//		dbvector oneRow(colsNum);
//		this->mMat.push_back(oneRow);
//
//		for (uint j = 0; j < colsNum; j++)
//		{
//			this->mMat[i][j] = mat[i][j];
//		}
//	}
//}

// Destructor
MMatrix::~MMatrix()
{
	for (uint i = 0; i < this->rowsNum; i++)
	{
		this->mMat[i].clear();
	}
	this->mMat.clear();
}

// Get number of rows
size_t MMatrix::getRowsNum() const
{
	return this->rowsNum;
}

// Get number of columns
size_t MMatrix::getColsNum() const
{
	return this->colsNum;
}

// Set number of rows
void MMatrix::setRowsNum(size_t m)
{
	this->rowsNum = m;
}

// Set number of columns
void MMatrix::setColsNum(size_t n)
{
	this->colsNum = n;
}

// set element at i, j of a value
void MMatrix::setElement(uint i, uint j, double val)
{
	this->mMat[i][j] = val;
}

// set a block from rInit to rEnd and cInit to cEnd with a value
// blockRange(rInit, rEnd, cInit, cEnd)
void MMatrix::setBlock(double val, Range2D blockRange)
{
	int height = this->rowsNum;
	int width = this->colsNum;

	int rInit, rEnd, cInit, cEnd;

	if (std::get<1>(blockRange) >= height)
	{
		std::cerr << "Warning: 'rEnd' out of matrix boundary" << std::endl;
		rEnd = height;
	}
	else if (std::get<1>(blockRange) <= 0)
	{
		rEnd = int(height) + std::get<1>(blockRange);
		if (rEnd <= 0) rEnd = height;
	}
	else
	{
		rEnd = std::get<1>(blockRange) + 1;
	}

	if ((std::get<0>(blockRange) >= rEnd) || (std::get<0>(blockRange) < 0))
	{
		std::cerr << "Warning: illegal 'rInit' value" << std::endl;
		rInit = 0;
	}
	else
	{
		rInit = std::get<0>(blockRange);
	}

	if (std::get<3>(blockRange) >= width)
	{
		std::cerr << "Warning: 'cEnd' out of matrix boundary" << std::endl;
		cEnd = width;
	}
	else if (std::get<3>(blockRange) <= 0)
	{
		cEnd = int(width) + std::get<3>(blockRange);
		if (cEnd <= 0) cEnd = width;
	}
	else
	{
		cEnd = std::get<3>(blockRange) + 1;
	}

	if ((std::get<2>(blockRange) >= cEnd) || (std::get<2>(blockRange) < 0))
	{
		std::cerr << "Warning: illegal 'cInit' value" << std::endl;
		cInit = 0;
	}
	else
	{
		cInit = std::get<2>(blockRange);
	}

	for (int i = rInit; i < rEnd; i++)
	{
		for (int j = cInit; j < cEnd; j++)
		{
			this->mMat[i][j] = val;
		}
	}
}

// Copy a 'height' - by - 'width' block from source matrix 'srcMat' to current matrix
// Optional inital indeices for both source and target matrix
// blockRange(rInit_src, cInit_src, rInit_tar, cInit_tar)
void MMatrix::copyBlock(const MMatrix& srcMat, int height, int width,
	Range2D blockRange)
{
	int rInit_src, cInit_src, rInit_tar, cInit_tar;

	// Check validness of 'rInit_src'
	if ((std::get<0>(blockRange) + height) > srcMat.rowsNum )
	{
		std::cerr << "Warning: 'rInit_src + height' out of matrix boundary" << std::endl;
	}
	else if (std::get<0>(blockRange) < 0)
	{
		rInit_src = 0;
	}
	else
	{
		rInit_src = std::get<0>(blockRange);
	}

	// Check validness of 'cInit_src'
	if ((std::get<1>(blockRange) + width) > srcMat.colsNum)
	{
		std::cerr << "Warning: 'cInit_src + width' out of matrix boundary" << std::endl;
	}
	else if (std::get<1>(blockRange) < 0)
	{
		cInit_src = 0;
	}
	else
	{
		cInit_src = std::get<1>(blockRange);
	}

	// Check validness of 'rInit_tar'
	if ((std::get<2>(blockRange) + height) > this->rowsNum)
	{
		std::cerr << "Warning: 'rInit_tar + height' out of matrix boundary" << std::endl;
	}
	else if (std::get<2>(blockRange) < 0)
	{
		rInit_tar = 0;
	}
	else
	{
		rInit_tar = std::get<2>(blockRange);
	}

	// Check validness of 'cInit_tar'
	if ((std::get<3>(blockRange) + width) > this->colsNum)
	{
		std::cerr << "Warning: 'cInit_tar + width' out of matrix boundary" << std::endl;
	}
	else if (std::get<3>(blockRange) < 0)
	{
		cInit_tar = 0;
	}
	else
	{
		cInit_tar = std::get<3>(blockRange);
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			this->mMat[rInit_tar + i][cInit_tar + j] 
		 = srcMat.mMat[rInit_src + i][cInit_src + j];
		}
	}
}

double MMatrix::getElement(uint i, uint j)
{
	return this->mMat[i][j];
}

// Matrix operation: assignment (=)
MMatrix& MMatrix:: operator= (const MMatrix& assigned)
{
	this->rowsNum = assigned.getRowsNum();
	this->colsNum = assigned.getColsNum();

	this->mMat.resize(rowsNum);

	for (uint i = 0; i < this->rowsNum; i++)
	{
		this->mMat[i].resize(colsNum);
		for (uint j = 0; j < this->colsNum; j++)
		{
			this->mMat[i][j] = assigned.mMat[i][j];
		}
	}
	return (*this);
}

// Matrix operation: equal to (==)
bool MMatrix:: operator== (const MMatrix& compared)
{
	bool res = true;

	for (uint i = 0; i < this->rowsNum; i++)
	{
		res = ( res && (this->mMat[i] == compared.mMat[i]) );
	}
	return res;
}

// Matrix operation: element-wise greater than (>)
MMatrix MMatrix:: operator>(const MMatrix& compared)
{
	// Initialize the resulted matrix
	MMatrix resMat(this->rowsNum, this->colsNum, 0.0);

	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			resMat.mMat[i][j] = (this->mMat[i][j] > compared.mMat[i][j]);
		}
	}
	return resMat;
}

// Matrix operation: greater than threshold (>)
MMatrix MMatrix:: isGreator(double thres)
{
	// Initialize the resulted matrix
	MMatrix resMat(this->rowsNum, this->colsNum, 0.0);

	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			resMat.mMat[i][j] = (this->mMat[i][j] > thres);
		}
	}
	return resMat;
}

// Matrix operation: add
MMatrix& MMatrix:: operator+= (const MMatrix& added)
{
	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			this->mMat[i][j] += added.mMat[i][j];
		}
	}
	return (*this);
}

// Matrix operation: add (two matrix)
MMatrix MMatrix:: operator+(const MMatrix& added)
{
	// Initialize the resulted matrix
	MMatrix resMat(this->rowsNum, this->colsNum, 0.0);

	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			resMat.mMat[i][j] = (this->mMat[i][j] + added.mMat[i][j]);
		}
	}
	return resMat;
}

// Matrix operation: added by single number
void MMatrix:: add(double added)
{
	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			this->mMat[i][j] += added;
		}
	}
}

// Matrix operation: subtract
MMatrix& MMatrix:: operator-= (const MMatrix& subtracted)
{
	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			this->mMat[i][j] -= subtracted.mMat[i][j];
		}
	}
	return (*this);
}

// Matrix operation: subtract (two matrix)
MMatrix MMatrix:: operator-(const MMatrix& subtracted)
{
	// Initialize the resulted matrix
	MMatrix resMat(this->rowsNum, this->colsNum, 0.0);

	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			resMat.mMat[i][j] = (this->mMat[i][j] - subtracted.mMat[i][j]);
		}
	}
	return resMat;
}

// Matrix operation: subtracted by single number
void MMatrix:: sub(double subtracted)
{
	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			this->mMat[i][j] -= subtracted;
		}
	}
}

// Matrix operation: element-wise multiplication
MMatrix& MMatrix:: operator*=(const MMatrix& multiplied)
{
	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			this->mMat[i][j] *= multiplied.mMat[i][j];
		}
	}
	return (*this);
}

// Matrix operation: element-wise multiplication (.*)
MMatrix MMatrix:: times(const MMatrix& multiplied)
{
	// Initialize the resulted matrix
	MMatrix resMat(this->rowsNum, this->colsNum, 0.0);

	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			resMat.mMat[i][j] = (this->mMat[i][j] * multiplied.mMat[i][j]);
		}
	}
	return resMat;
}

// Matrix operation: multiplied by single number
void MMatrix:: mul(double multiplied)
{
	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			this->mMat[i][j] *= multiplied;
		}
	}
}

// Matrix operation: inner product (matrix multiplication)
MMatrix MMatrix:: operator* (const MMatrix& multiplied)
{
	// Get size of multiplied matrix (number of rows and columns)
	size_t multipliedRowsNum = multiplied.getRowsNum();
	size_t multipliedColsNum = multiplied.getColsNum();

	if (this->colsNum != multipliedRowsNum)
	{
		std::cout << 
	"Error - column number of first matrix and row number of the second one are not equal!" 
			<< std::endl;

		exit(EXIT_FAILURE);
	}
	else
	{
		// Initialize the resulted matrix
		MMatrix resMat(this->rowsNum, multipliedColsNum, 0.0);

		// inner product by $resMat_{ij} = \sum{k=1}{colsNum}mMat_{ik}\times multiplied_{kj}$
		for (uint i = 0; i < this->rowsNum; i++)
		{
			for (uint j = 0; j < multipliedColsNum; j++)
			{
				for (uint k = 0; k < multipliedRowsNum; k++)
				{
					resMat.mMat[i][j] += this->mMat[i][k] * multiplied.mMat[k][j];
				}			
			}
		}
		return resMat;
	}
}

// Matrix operation: element-wise divide : divided by a matrix
MMatrix& MMatrix:: operator/=(const MMatrix& divided)
{
	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			if (divided.mMat[i][j] != 0) // To avoid inf
			{
				this->mMat[i][j] /= divided.mMat[i][j];
			}			
		}
	}
	return (*this);
}

// Matrix operation: element-wise divide (/)
MMatrix MMatrix:: operator/(const MMatrix& divided)
{
	// Initialize the resulted matrix
	MMatrix resMat(this->rowsNum, this->colsNum, 0.0);

	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			if (divided.mMat[i][j] != 0) // To avoid inf
			{
				resMat.mMat[i][j] = (this->mMat[i][j] / divided.mMat[i][j]);
			}
		}
	}
	return resMat;
}

// Matrix operation: divided by single number
void MMatrix::div(double divided)
{
	if (divided == 0)
	{
		std::cout <<
			"Error - cannot divided by 0"
			<< std::endl;

		exit(EXIT_FAILURE);
	}
	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			this->mMat[i][j] /= divided;
		}
	}
}

// Matrix operation: logarithm
void MMatrix::loga(double base)
{
	if (base == NULL)
	{
		for (uint i = 0; i < this->rowsNum; i++)
		{
			for (uint j = 0; j < this->colsNum; j++)
			{
				this->mMat[i][j] = log(this->mMat[i][j]);
			}
		}
	}
}

// Matrix operation: square root
void MMatrix::sqroot(void)
{
	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			this->mMat[i][j] = sqrt(this->mMat[i][j]);
		}
	}
}

// Matrix operation: transform
MMatrix MMatrix:: operator~()
{
	// Initialize the resulted matrix
	MMatrix resMat(this->colsNum, this->rowsNum);

	for (uint i = 0; i < this->colsNum; i++)
	{
		for (uint j = 0; j < this->rowsNum; j++)
		{
				resMat.mMat[i][j] += this->mMat[j][i];

		}
	}
	return resMat;
}

// Matrix operation: truncation
MMatrix MMatrix:: truncate(int row0, int row1, int col0, int col1)
{
	// If negative one, then point to first row
	if (row0 < 0) row0 = 0; 

	// If negative x, then point to last row + 1 - x
	if (row1 < 0) row1 += this->rowsNum;

	// If negative one, then point to first col
	if (col0 < 0) col0 = 0;

	// If negative x, then point to last col + 1 - x
	if (col1 < 0) col1 += this->colsNum;

	if ( (row1 < row0) || (row1 < 0) )
	{
		std::cout <<
			"Error - row index out of boundary"
			<< std::endl;

		exit(EXIT_FAILURE);
	}
	else if ((uint)row1 >= rowsNum)
	{
		std::cout <<
			"Error - row index out of boundary: too large"
			<< std::endl;

		exit(EXIT_FAILURE);
	}

	if ( (col1 < col0) || (col1 < 0) )
	{
		std::cout <<
			"Error - column index out of boundary"
			<< std::endl;

		exit(EXIT_FAILURE);
	}
	else if ((uint)col1 >= colsNum)
	{
		std::cout <<
			"Error - column index out of boundary: too large"
			<< std::endl;

		exit(EXIT_FAILURE);
	}

	// Size of the truncated matrix
	size_t newRowsNum = row1 - row0 + 1;
	size_t newColsNum = col1 - col0 + 1;

	// Initialize the truncated matrix
	MMatrix resMat(newRowsNum, newColsNum, 0.0);

	for (uint i = 0; i < newRowsNum; i++)
	{
		for (uint j = 0; j < newColsNum; j++)
		{
			resMat.mMat[i][j] = this->mMat[row0+i][col0+j];
		}
	}
	return resMat;
}

// Matrix operation: maximum value of whole matrix
double MMatrix::max()
{
	double maxVal = this->mMat[0][0];

	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			if (this->mMat[i][j] > maxVal)
			{
				maxVal = this->mMat[i][j];
			}
		}
	}
	return maxVal;
}

// Matrix operation: minimum value of whole matrix
double MMatrix::min()
{
	double minVal = this->mMat[0][0];

	for (uint i = 0; i < this->rowsNum; i++)
	{
		for (uint j = 0; j < this->colsNum; j++)
		{
			if (this->mMat[i][j] < minVal)
			{
				minVal = this->mMat[i][j];
			}
		}
	}
	return minVal;
}

// Display matrix in console
void MMatrix::display()
{
	for (uint i = 0; i < rowsNum; i++)
	{
		for (uint j = 0; j < colsNum; j++)
		{
			std::cout << mMat[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
// (Row) Vector class (inherit from Matrix class)
///////////////////////////////////////////////////////////////////////////////
//class MVector : public MMatrix  // Declared in "mathTool.h"
//{
//	public:
//		// Constructor : method 0 - initialize an empty vector
//		//MVector();
//
//		// Constructor : method 1 -initialize a matrix with n capacity
//		MVector(size_t n); 
//
//		// Constructor : method 2 -initialize a matrix with a value
//		MVector(size_t n, double initVal);
//
//		// Constructor : method 3 -convert an existing array to vector
//		//MVector(double* arr);
//
//		// Destructor
//		~MVector();
//
//		// Get matrix size (number of rows or columns)
//		size_t getLength() const;
//
//		// set element at j of a value
//		void setElement(uint j, double val);
//
//		// get value of element at j
//		double getElement(uint j);
//
//		//------------------------------------------------------------------
//
//		// append a value at the end of the vector
//		void append(double appendVal);
//
//		// Insert a value at a position
//		void insert(uint posi, double insertVal);
//};

// Constructor : method 0
// MVector::MVector() : MMatrix(1) {} // Not functioning,  needed to be updated in future 

// Constructor : method 1
MVector::MVector(size_t n) : MMatrix(1, n) {}

// Constructor : method 2
MVector::MVector(size_t n, double initVal) : MMatrix(1, n, initVal) {}
 
// Constructor : method 3  //  needed to be updated in future 
//MVector::MVector(double* arr) : MMatrix(1)
//{
//	for (uint i = 0; i < this->getColsNum(); i++)
//	{
//		this->mMat[0].push_back(arr[i]);
//		std::cout << "...here...";
//	}
//}

// Destructor
MVector::~MVector() {}

// Get number of rows
size_t MVector::getLength() const
{
	return this->getColsNum();
}

// set element at j of a value
void MVector::setElement(uint j, double val)
{
	this->mMat[0][j] = val;
}

// set value of element at j
double MVector::getElement(uint j)
{
	return this->mMat[0][j];
}

// append a value at the end of the vector
void MVector::append(double appendVal)
{
	this->mMat[0].push_back(appendVal);

	this->setColsNum( this->mMat[0].size() ); // update vector length
}

// insert a value at a position of the vector // Improvement needed
void MVector::insert(uint posi, double insertVal)
{
	// utilize a temporal vector
	dbvector tempVector = this->mMat[0]; 
	dbiterator itr = tempVector.begin() + posi;
	
	tempVector.insert(itr, insertVal); // insert in the temporal vector

	this->mMat[0].clear();  // Is this necessary ? 11/28/2015
	this->mMat.clear();		// Is this necessary ? 11/28/2015

	this->mMat.push_back(tempVector); // append the temporal vector

	tempVector.clear();  // clear the temporal vector

	this->setColsNum( this->mMat[0].size() ); // update vector length
}


///////////////////////////////////////////////////////////////////////////////
// Other Mathematical functions // Declared in "mathTool.h"
///////////////////////////////////////////////////////////////////////////////

MVector pascalTriangle(size_t winSize, double initVal1, double initVal2)
{
	MVector retMat(winSize, 0.0);

	if (winSize > 2)
	{
		MVector initialMat(2, 0.0); 

		// set initial values 
		initialMat.setElement(0, initVal1);
		initialMat.setElement(1, initVal2);

		// return matrix assigned to be [1, 1]
		retMat = initialMat;

		for (uint i = 2; i <= winSize-1; i++)
		{
			// assigned temporal matrix to be last return matrix
			MVector tempMat(winSize, 0.0);
			tempMat = retMat;

			// return matrix assigned to be [1, 1] again
			retMat = initialMat; 

			for (uint j = 1; j < i; j++)
			{ 
				double insertVal = 
					(tempMat.getElement(j) + tempMat.getElement(j - 1));

				retMat.insert(j, insertVal);		
			}
		}
	}
	return retMat;
}