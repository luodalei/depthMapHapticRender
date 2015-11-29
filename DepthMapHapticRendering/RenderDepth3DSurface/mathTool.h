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

#include <iostream>
#include <vector>
//------------------------------------------------------------------------------
typedef unsigned int uint; // unsigned integer
typedef std::vector<double> dbvector; // vector of double type
typedef std::vector<double>::iterator dbiterator; // iterator of double type;
typedef std::vector< std::vector< double > > dbmatrix; // matrix of double type

													   
													   
///////////////////////////////////////////////////////////////////////////////
// Matrix class
///////////////////////////////////////////////////////////////////////////////

class MMatrix
{
private:
	// matrix with size of rowsNum $\times$ colsNum
	size_t rowsNum; // number of rows
	size_t colsNum; // number of columns

protected:
	dbmatrix mMat; // the matrix

public:
	// Constructor : method 0 -initialize only the row vector
	//MMatrix(size_t m); // Not functioning currently

	// Constructor : method 1 -initialize a matrix with m $\times$ n capacity
	MMatrix(size_t m, size_t n);

	// Constructor : method 2 -initialize a matrix with a value
	MMatrix(size_t m, size_t n, double initVal);

	// Constructor : method 3 -convert an existing 2D array to matrix
	//MMatrix(double** mat);

	// Destructor
	~MMatrix();

	// Get matrix size (number of rows or columns)
	size_t getRowsNum() const;
	size_t getColsNum() const;

	// Set matrix size (number of rows or columns)
	void setRowsNum(size_t m);
	void setColsNum(size_t n);

	// set element at i, j of a value
	void setElement(uint i, uint j, double val);

	// get value of element at i, j
	double getElement(uint i, uint j);

	//------------------------------------------------------------------
	// Matrix operation: assignment (=) 
	// Matrix dimension may subject to change
	MMatrix& operator= (const MMatrix& assigned);

	// Matrix operation: equal to (==)
	bool operator== (const MMatrix& compared);

	// Matrix operation: add (+=)
	MMatrix& operator+= (const MMatrix& added);

	// Matrix operation: subtract (-=)
	MMatrix& operator-= (const MMatrix& subtracted);

	// Matrix operation: element-wise multiplication (.*)
	MMatrix& operator*=(const MMatrix& multiplied);

	// Matrix operation: inner product (*) 
	MMatrix operator* (const MMatrix& multiplied);

	// Matrix operation: element-wise divide (./)
	MMatrix& operator/=(const MMatrix& divided);

	// Matrix operation: transform (')
	MMatrix operator~();

	// Matrix operation: truncation
	MMatrix truncate(int row0, int row1, int col0, int col1);
	//------------------------------------------------------------------

	//  Display matrix in console (unsuitable for large matrix)
	void display();
};

///////////////////////////////////////////////////////////////////////////////
// (Row) Vector class (inherit from Matrix class)
///////////////////////////////////////////////////////////////////////////////

class MVector : public MMatrix
{
public:
	// Constructor : method 0 - initialize an empty vector
	MVector();

	// Constructor : method 1 -initialize a matrix with n capacity
	MVector(size_t n);

	// Constructor : method 2 -initialize a matrix with a value
	MVector(size_t n, double initVal);

	// Constructor : method 3 -convert an existing array to vector
	MVector(double* arr);

	// Destructor
	~MVector();

	// Get matrix size (number of rows or columns)
	size_t getLength() const;

	// set element at j of a value
	void setElement(uint j, double val);

	// get value of element at j
	double getElement(uint j);

	//------------------------------------------------------------------

	// append a value at the end of the vector
	void append(double appendVal);

	// Insert a value at a position
	void insert(uint posi, double insertVal);
};

///////////////////////////////////////////////////////////////////////////////
// Other Mathematical functions
///////////////////////////////////////////////////////////////////////////////

MVector pascalTriangle(size_t winSize, double initVal1, double initVal2);