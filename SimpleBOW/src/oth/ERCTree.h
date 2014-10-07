#ifndef __ERCTree_h__
#define __ERCTree_h__

#include <iostream>
#include <string>
#include <vector>
#include <climits>

#include "InfMethods.h"

class TCodeBookBinaryTreeNode{

public:
	const TCodeBookBinaryTreeNode *parrent;
	TCodeBookBinaryTreeNode *left, *right;
	int leftID, rightID;
	bool status;


	int getRightmostID( ){
		if( right == NULL){
			return rightID;
		} else {
			return right->getRightmostID();
		}
	}

	TCodeBookBinaryTreeNode( const TCodeBookBinaryTreeNode *_parrent)
		: parrent( _parrent), left( NULL), right( NULL), leftID( 0), rightID( 0), status( false)
	{
	}

	TCodeBookBinaryTreeNode( std::istream &stream, const TCodeBookBinaryTreeNode *_parrent, const bool bin = false)
		: parrent( _parrent), left( NULL), right( NULL), leftID( 0), rightID( 0), status( false)
	{
		if( bin){
			stream.read( (char *) &left, sizeof( TCodeBookBinaryTreeNode *));
			stream.read( (char *) &right, sizeof( TCodeBookBinaryTreeNode *));
			stream.read( (char *) &leftID, sizeof( int));
			stream.read( (char *) &rightID, sizeof( int));
		} else {
			stream >> *(int *)&left >> *(int *)&right >> leftID >> rightID;
		}
	}

	virtual void write( std::ostream &stream, const bool binary = false) const{
		if( binary){
			stream.write( (const char *) &left, sizeof( TCodeBookBinaryTreeNode *));
			stream.write( (const char *) &right, sizeof( TCodeBookBinaryTreeNode *));
			stream.write( (const char *) &leftID, sizeof( int));
			stream.write( (const char *) &rightID, sizeof( int));
		} else {
			stream << (bool)( left != NULL) << ' ' << (bool)( right != NULL) << ' ';
			stream << leftID << ' ' << rightID << ' '; 
		}
	};

	bool OK() const
	{
		return status;
	}

	int getWord( const float * point) const
	{
		const bool goLeftResult = this->goLeft( point);
		if( left == NULL && goLeftResult){
			return leftID;
		} else if( right == NULL && !goLeftResult){
			return rightID;
		} else if( goLeftResult){
			return left->getWord( point);
		} else {
			return right->getWord( point);
		}
	};

	virtual bool goLeft( const float *point) const = 0;

	
	int assignIDs( int lastID){

		if( left == NULL){
			leftID = lastID++;
		} else {
			lastID = left->assignIDs( lastID);
		}
		
		if( right == NULL){
			rightID = lastID++;
		} else {
			lastID = right->assignIDs( lastID);
		}

		return lastID;
	};

	virtual ~TCodeBookBinaryTreeNode(){
		delete left;
		delete right;
	}

};



class TERCTreeNode : public TCodeBookBinaryTreeNode{

	static const std::string filePrefix;

public:
	int decisionDimension;
	float decisionThreshold;

	TERCTreeNode( const std::vector< float *>& data, const int dataDim, const std::vector< int>& labels, const double Smin, const int Tmax, const TCodeBookBinaryTreeNode *_parrent, const int levelsLeft = INT_MAX, const bool randomThreshold = true, const TInformaionComputer method = TInformaionComputer());
	TERCTreeNode( std::istream &stream, const TCodeBookBinaryTreeNode *_parrent);

	virtual void write( std::ostream &, const bool bin = true) const;

	virtual bool goLeft( const float *point) const;

	virtual ~TERCTreeNode(){
		this->TCodeBookBinaryTreeNode::~TCodeBookBinaryTreeNode();
	}

	static bool read( const std::string fileName, std::vector< TERCTreeNode *> &trees);
	static bool write( const std::string fileName, std::vector< TERCTreeNode *> &trees);
};

class TCMPTreeNode : public TCodeBookBinaryTreeNode{

	static const std::string filePrefix;

public:
	int decDim1, decDim2;

	TCMPTreeNode( const std::vector< float *>& data, const int dataDim, const std::vector< int>& labels, const int Tmax, const TCodeBookBinaryTreeNode *_parrent, const int levelsLeft = INT_MAX, const TInformaionComputer method = TInformaionComputer());
	TCMPTreeNode( std::istream &stream, const TCodeBookBinaryTreeNode *_parrent);

	virtual void write( std::ostream &, const bool bin = true) const;

	virtual bool goLeft( const float *point) const;

	virtual ~TCMPTreeNode(){
		this->TCodeBookBinaryTreeNode::~TCodeBookBinaryTreeNode();
	}

	static bool read( const std::string fileName, std::vector< TCMPTreeNode *> &trees);
	static bool write( const std::string fileName, std::vector< TCMPTreeNode *> &trees);
};

class TPCTreeNode : public TCodeBookBinaryTreeNode{

	static std::vector<float> normalDistribution;
	float getNormalDistrSideIntegral( const float point) const;

	static const std::string filePrefix;
	static const std::string filePrefixBin;


public:
	std::vector< float> projection;
	float threshold;
	float standardDeviation;

	TPCTreeNode( const std::vector< float *>& data, const int dataDim, std::vector< int>& labels, 
		const double Smin, const int Tmax, 
		const TCodeBookBinaryTreeNode *_parrent, 
		const int levelsLeft = INT_MAX, 
		const bool randomThreshold = true, 
		const TInformaionComputer &method = TInformaionComputer(), 
		const std::string &projectionSelectionMethod = "RANDOM", 
		const int samplesToEstimateCovariance = 512);
	TPCTreeNode( std::istream &stream, const TCodeBookBinaryTreeNode *_parrent, const bool binary);

	virtual void write( std::ostream &, const bool bin = false) const;

	virtual bool goLeft( const float *point) const;

	virtual ~TPCTreeNode(){
		this->TCodeBookBinaryTreeNode::~TCodeBookBinaryTreeNode();
	}

	void getWords( const float * point, std::vector< int>& words) const;


	static bool read( const std::string fileName, std::vector< TPCTreeNode *> &trees);
	static bool write( const std::string fileName, std::vector< TPCTreeNode *> &trees);
};

#endif