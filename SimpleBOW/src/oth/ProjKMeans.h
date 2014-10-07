#ifndef __ProjKMeans_h__
#define __ProjKMeans_h__

#include <iostream>
#include <string>
#include <vector>
#include <climits>

#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include <opencv2/legacy/legacy.hpp>

class TProjKmeans{

	static const std::string firstLineString;
	static const int samplesToEstimateCovariance;
	static const bool normDifferences;
	static const bool myCovariance;


	CvMat *projection;
	CvMat *centers;
	CvMat *projectedPoint;
	CvFeatureTree *tree;

	double avgWordDistance;

	bool exactKNN;

	void showData( CvMat *projectedData, const std::vector< int> &labels);

	void generateRandomData( const std::vector< float *>& data, const int dimension);

	void initLDP( const std::vector< float *>& data, const int dimension, CvMat *projection, const int words, 
			 const int samplesToEstimateCovariance,
			 const bool normDifferences = false, const bool myCovariance = true);

	void projectData( const std::vector< float *>& data, CvMat *projection, CvMat *projectedData);

    bool statusOK;

	int inputVectorDimension;

public:

	TProjKmeans( std::string filename);

	TProjKmeans( const std::vector< float *>& data, const int dimension, const int words, const int maxIteration, const bool exactKNN = false);
	TProjKmeans( const std::vector< float *>& data, const int dimension, const int targetDim, const int words, const int maxIteration, const bool exactKNN = false);

	TProjKmeans( const std::vector< float *>& data, const int dimension, const std::vector< int> labels,  const int targetDim, const int words, const int maxIteration, const bool PCA = false, const bool exactKNN = false);

	void write( std::string filename);

	int getWord( const float *point) const;
	void getWord( const float *point, std::vector< int> &nn) const;
	void getWord( const float *point, std::vector< int> &nn, std::vector< double> &distance) const;

	int getInputDimension() const { return inputVectorDimension;}

	int getWordCount() const{
		if( centers != NULL){
			return centers->rows;
		} else {
			return 0;
		}
	}

	~TProjKmeans(){
		if( projection != NULL)     cvReleaseMat( &projection);
		if( centers != NULL)        cvReleaseMat( &centers);
		if( projectedPoint != NULL) cvReleaseMat( &projectedPoint);
		if( tree != NULL)           cvReleaseFeatureTree( tree);
	}

    bool OK(){
        return statusOK;
    }

	double getAvgWordDistance( const int estimateCount = 10000);

};

#endif
