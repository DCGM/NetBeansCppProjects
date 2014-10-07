#include  "InfMethods.h"

#include <map>
#include <cmath>
#include <algorithm>
#include <iostream>


using namespace std;


void LDP( const vector< float *>& allData, const int dimension, const vector< int> &allLabels, CvMat * projections, CvMat * resultEigenValues, const int samplesToEstimateCovariance, const bool normDifferences, const bool myCovariance, const int classesToUse)
{

	assert( allData.size() == allLabels.size());
	assert( projections != NULL);
	assert( projections->cols == dimension);
	assert( resultEigenValues == NULL || projections->rows == resultEigenValues->cols && resultEigenValues->rows == 1);

	cvSetZero( projections);
	cvSetZero( resultEigenValues);

	vector< float *> sampledData;
	vector< int> sampledLabels;

	if( classesToUse > 1){
		
		map< int, int> sortedLabels;
		for( int i = 0; i < (int)allLabels.size(); i++){
			sortedLabels[ allLabels[ i]]++;
		}

		vector< int> labels;
		for( map< int, int>::const_iterator it = sortedLabels.begin(); it != sortedLabels.end(); it++){
			labels.push_back( it->first);
		}

		while( (int)labels.size() > classesToUse){
			labels[ rand() % labels.size()] = labels.back();
			labels.pop_back();
		}

		sortedLabels.clear();
		for( int i = 0; i < (int) labels.size(); i++){
			sortedLabels[ labels[ i]] = 1;
		}

		for( int i = 0; i < (int)allData.size(); i++){
			if( sortedLabels.find( allLabels[ i]) != sortedLabels.end()){
				sampledData.push_back( allData[ i]);
				sampledLabels.push_back( allLabels[ i]);
			}
		}
	}

	const vector< float *> &data = (classesToUse > 0)?( sampledData):( allData);
	const vector< int> &labels = (classesToUse > 0)?( sampledLabels):( allLabels);

	// sort samples according to their label
	map< int, vector< int> > sortedLabel;
	for( int i = 0; i < (int)labels.size(); i++){
		sortedLabel[ labels[ i]].push_back( i);
	}

	if( sortedLabel.size() < 2){
		return;
	}

	CvMat * samplesForCovariance = cvCreateMat( samplesToEstimateCovariance, dimension, CV_64F);
	CvMat* avg = cvCreateMat( 1, dimension, CV_64F);
	cvSet( avg, cvScalarAll( 0.0));

	// ===================================================================
	// compute Cd - the covariance matrix of differently labeled samples
	// prepare stirred list of samples (and their labels)
	vector< int> samplesLeft( labels.size());
	for( int i = 0; i < (int)samplesLeft.size(); i++){
		samplesLeft[ i] = i;
	}

	// randomize the sample order
	vector< int> randomizedSamples( labels.size());
	for( int i = 0; i < (int)randomizedSamples.size(); i++){
		const int randID = rand() % samplesLeft.size();
		randomizedSamples[ i] = samplesLeft[ randID];
		samplesLeft[ randID] = samplesLeft.back();
		samplesLeft.pop_back();
	}

	// compute differences of random samples from different classes
	for( int i = 0; i < samplesToEstimateCovariance; i ++){

		//select first sample
		const int firstSampleID = rand() % labels.size();

		// for now, this indexes randomizedSamples
		int secondSampleID = rand() % randomizedSamples.size();

		// search for first sample from different class 
		// searching in randomizedSamples, so this does not change the probability distribution
		while( labels[ firstSampleID] == labels[ randomizedSamples[ secondSampleID]]){
			secondSampleID = (secondSampleID + 1) % randomizedSamples.size();
		}
		// now indexing samples directly
		secondSampleID = randomizedSamples[ secondSampleID];
		
		// subtract the two selected samples and put them to samplesForCovariance
		double *ptr = (double *) cvPtr2D( samplesForCovariance, i, 0);
		for( int j = 0; j < dimension; j++){
			ptr[ j] = data[ firstSampleID][ j] - data[ secondSampleID][ j];
		}
	}

	if( normDifferences){
		CvMat *longRow = cvCreateMatHeader( 1, samplesForCovariance->cols, samplesForCovariance->type);
		for( int i = 0; i < samplesToEstimateCovariance; i ++){
			cvGetRow( samplesForCovariance, longRow, i);
			cvScale( longRow, longRow, 1.0 / max( cvNorm( longRow), 1e-5));
		}
		cvReleaseMat( &longRow);
	}

	// Compute Cd - covariance of between class differences
	CvMat* Cd = cvCreateMat( samplesForCovariance->cols, samplesForCovariance->cols, CV_64F);

	if( myCovariance){
		myCalcCovar( samplesForCovariance, Cd);
	} else {
		cvCalcCovarMatrix( (const CvArr **) &samplesForCovariance, 1, Cd, avg, CV_COVAR_ROWS | CV_COVAR_NORMAL);
	}

	// compute Cd - the covariance matrix of differently labeled samples
	// ===================================================================


	// ===================================================================
	// compute Cs - the covariance matrix of samples with same label
	
	CvMat* Cs = cvCreateMat( samplesForCovariance->cols, samplesForCovariance->cols, CV_64F);
	cvSetIdentity( Cs);

	bool samplesGood = false;
	for( map< int, vector< int> >::const_iterator it = sortedLabel.begin(); it != sortedLabel.end(); it++){
		samplesGood |= it->second.size() > 1;
	}
	if( samplesGood){

		// compute differences of random samples from the same class
		for( int i = 0; i < samplesToEstimateCovariance; i ++){

			// select first sample from class which has at least 2 samples
			int firstSampleID = rand() % labels.size();
			while( sortedLabel[ labels[ firstSampleID] ].size() < 2){
				firstSampleID = rand() % labels.size();
			}

			const vector< int> &sameLabelSamples = sortedLabel[ labels[ firstSampleID]];

			// select second sample which is not the same as the first one
			int secondSampleID = sameLabelSamples[ rand() % sameLabelSamples.size()];
			while( secondSampleID == firstSampleID){
				secondSampleID = sameLabelSamples[ rand() % sameLabelSamples.size()];
			}

			// subtract the two selected samples and put them to samplesForCovariance
			double *ptr = (double *) cvPtr2D( samplesForCovariance, i, 0);
			for( int j = 0; j < dimension; j++){
				ptr[ j] = data[ firstSampleID][ j] - data[ secondSampleID][ j];
			}
		}

		if( normDifferences){
			CvMat *longRow = cvCreateMatHeader( 1, samplesForCovariance->cols, samplesForCovariance->type);
			for( int i = 0; i < samplesToEstimateCovariance; i ++){
				cvGetRow( samplesForCovariance, longRow, i);
				cvScale( longRow, longRow, 1.0 / max( cvNorm( longRow), 1e-5));
			}
			cvReleaseMat( &longRow);
		}
	}


	// Compute Cs - covariance of differences between samples from the same class
	if( myCovariance){
		myCalcCovar( samplesForCovariance, Cs);
	} else {
		cvCalcCovarMatrix( (const CvArr **) &samplesForCovariance, 1, Cs, avg, CV_COVAR_ROWS | CV_COVAR_NORMAL);
	}

	// compute Cs - the covariance matrix of samples with same label
	// ===================================================================


	// ===================================================================
	// compute the transformations as U = eig( Cs^-1 Cd)

	// invert Cs
	cvInvert( Cs, Cs, CV_SVD_SYM);

	// multiply Cs^-1 Cd
	cvGEMM( Cs, Cd, 1.0, NULL, 1.0, Cs);

	// compute eigen vectors of  Cs^-1 Cd
	CvMat* eigenValues = cvCreateMat( 1, Cs->cols, CV_64F);
	CvMat* eigenVectorsMat = cvCreateMat( Cs->cols, Cs->cols, CV_64F);

	cvEigenVV	( Cs, eigenVectorsMat, eigenValues);

	// reverse order of the eigen values to match order of eigen vectors (there is a bug in OpenCV 2.0 which gives eigen vectors in correct order, but the eigen values are reversed)
	if( cvGetReal1D( eigenValues, 0) < cvGetReal1D( eigenValues, eigenValues->cols - 1)){
		cvFlip( eigenValues, 0, 1); 
	}

	// compute the transformations as U = eig( Cs^-1 Cd)
	// ===================================================================

	{
		CvMat * subMat = cvCreateMatHeader( projections->rows, projections->cols, eigenVectorsMat->type);
		cvGetRows( eigenVectorsMat, subMat, 0, projections->rows);
		cvConvert( subMat, projections);
		cvReleaseMat( &subMat);
	}

	if( resultEigenValues != NULL){
		CvMat * subMat = cvCreateMatHeader( 1, resultEigenValues->cols, eigenValues->type);
		cvGetCols( eigenValues, subMat, 0, resultEigenValues->cols);
		cvConvert( subMat, resultEigenValues);
		cvReleaseMat( &subMat);
	}

	cvReleaseMat( &eigenVectorsMat);
	cvReleaseMat( &eigenValues);

	cvReleaseMat( &Cs);
	cvReleaseMat( &Cd);
	cvReleaseMat( &samplesForCovariance);
	cvReleaseMat( &avg);
}


void PCA( const vector< float *>& data, const int dimension, CvMat * &projections, CvMat * resultEigenValues, const int samplesToEstimateCovariance)
{
	assert( projections != NULL);
	assert( projections->cols == dimension);
	assert( resultEigenValues == NULL || projections->rows == resultEigenValues->cols && resultEigenValues->rows == 1);

	cvSetZero( projections);
	cvSetZero( resultEigenValues);

	CvMat * samplesForCovariance = cvCreateMat( samplesToEstimateCovariance, dimension, CV_64F);
	CvMat* avg = cvCreateMat( 1, dimension, CV_64F);
	cvSet( avg, cvScalarAll( 0.0));

	// ===================================================================
	// compute covariance matrix 

	// get random data
	for( int i = 0; i < samplesToEstimateCovariance; i ++){

		//select sample
		const int sampleID = rand() % data.size();

		// add it to the matrix
		double *ptr = (double *) cvPtr2D( samplesForCovariance, i, 0);
		for( int j = 0; j < dimension; j++){
			ptr[ j] = data[ sampleID][ j];
		}
	}

	// Compute C - covariance 
	CvMat* C = cvCreateMat( samplesForCovariance->cols, samplesForCovariance->cols, CV_64F);

	cvCalcCovarMatrix( (const CvArr **) &samplesForCovariance, 1, C, avg, CV_COVAR_ROWS | CV_COVAR_NORMAL | CV_COVAR_SCALE);

	// compute covariance matrix 
	// ===================================================================



	// compute eigen vectors
	CvMat* eigenValues = cvCreateMat( 1, C->cols, CV_64F);
	CvMat* eigenVectorsMat = cvCreateMat( C->cols, C->cols, CV_64F);
	
	cvEigenVV( C, eigenVectorsMat, eigenValues);
	
	// reverse order of the eigen values to match order of eigen vectors (there is a bug in OpenCV 2.0 which gives eigen vectors in correct order, but the eigen values are reversed)
	if( cvGetReal1D( eigenValues, 0) < cvGetReal1D( eigenValues, eigenValues->cols - 1)){
		cvFlip( eigenValues, 0, 1); 
	}

	cout << "Eigen Values ";
	for( int i = 0; i < eigenValues->cols; i++){
		cout << cvGetReal1D( eigenValues, i) << ' ';
	}
	cout << endl << endl;


	// reduce dimensionality (take only first n eigenvectors)
	{
		CvMat * subMat = cvCreateMatHeader( projections->rows, projections->cols, eigenVectorsMat->type);
		cvGetRows( eigenVectorsMat, subMat, 0,  projections->rows);
		cvConvert( subMat, projections);
		cvReleaseMat( &subMat);
	}

	if( resultEigenValues != NULL){
		CvMat * subMat = cvCreateMatHeader( 1, resultEigenValues->cols, eigenValues->type);
		cvGetCols( eigenValues, subMat, 0, resultEigenValues->cols);
		cvConvert( subMat, resultEigenValues);
		cvReleaseMat( &subMat);
	}
	
	cvReleaseMat( &eigenVectorsMat);
	cvReleaseMat( &eigenValues);

	cvReleaseMat( &C);
	cvReleaseMat( &samplesForCovariance);
	cvReleaseMat( &avg);
}


TInformaionComputer::TInformaionComputer( const std::string _method)
{
	if( _method == "Sc"){
		method = Sc;
	} else if( _method == "MutualInformation"){
		method = MutualInformation;
	} else if( _method == "Entropy"){
		method = Entropy;
	} else if( _method == "Balanced"){
		method = Balanced;
	} else if( _method == "ScWithError"){
		method = ScWithError;
	} else if( _method == "Mean"){
		method = Mean;
	} else {
		method = UNKNOWN;
		throw string( "Unknown information method: ") + _method;
	}
}


inline double log2( const double val)
{
	return log( val) / log( 2.0);
}

inline double H( const map< int, double> &classCounts, const double totalCount)
{
	double sum = 0.0;
	for( map< int, double>::const_iterator it = classCounts.begin(); it != classCounts.end(); it++){
		if( it->second != 0){
			sum -= it->second / totalCount * log2( it->second / totalCount);
		}
	}
	return sum;
}

double TInformaionComputer::evaluateThreshold( vector< TProjectedPoint> &data, const float &threshold) const
{

	if( method == ScWithError){

		sort( data.begin(), data.end(), TInformaionComputer::TProjectedPoint::cmp);

		const double totalCount = data.size();
		double leftCount = 0;
		double rightCount = totalCount; 

		map< int, double> classCounts;
		map< int, double> leftClassCounts;
		map< int, double> rightClassCounts;

		// get left and right classCounts
		for( int i = 0; i < (int) data.size(); i++){
			rightCount++;
			rightClassCounts[ data[ i].label]++;
		}

		double Ict = 0;
		double Ht = 0;
		const double Hc = H( classCounts, totalCount);

		// try thresholds
		for( int i = 0; i < (int) data.size(); i++){

			if( data[ i].val > threshold){
				return (2 * Ict) / ( Hc + Ht);
			}

			const int label = data[ i].label;

			// !!! This is absolutely nonsence, but it provides good results - the computation of Ict is wrong !!!!
			if( leftClassCounts[ label] > 0.1){
				Ict += - leftCount / totalCount * leftClassCounts[ label] / leftCount * log2( leftClassCounts[ label] / leftCount);
			}
			Ict += - rightCount / totalCount * rightClassCounts[ label] / rightCount * log2( rightClassCounts[ label] / rightCount);

			leftCount++;
			leftClassCounts[ label]++;
			rightCount--;
			rightClassCounts[ label]--;

			// update Ht 
			if( leftCount != 0 && rightCount != 0){
				Ht = -( leftCount / totalCount * log2( leftCount / totalCount) 
					 + rightCount / totalCount * log2( rightCount / totalCount));
			} else {
				Ht = 0.0;
			}

			// there is now way that leftClassCounts[ label] could be 0 - there is the sample we are currently passing
			Ict -= - leftCount / totalCount * leftClassCounts[ label] / leftCount * log2( leftClassCounts[ label] / leftCount);
			if( rightClassCounts[ label] != 0){
				Ict -= - rightCount / totalCount * rightClassCounts[ label] / rightCount * log2( rightClassCounts[ label] / rightCount);
			}

		}
	} 


	const double totalCount = data.size();
	double leftCount = 0;
	double rightCount = 0; 

	map< int, double> classCounts;
	map< int, double> leftClassCounts;
	map< int, double> rightClassCounts;

	// get left and right classCounts
	for( int i = 0; i < (int) data.size(); i++){
		if( data[ i].val > threshold){
			rightCount++;
			rightClassCounts[ data[ i].label]++;
		} else {
			leftCount++;
			leftClassCounts[ data[ i].label]++;
		}
	}

	// get classCounts
	classCounts = rightClassCounts;
	for( map< int, double>::const_iterator it = leftClassCounts.begin(); it != leftClassCounts.end(); it++){
		classCounts[ it->first] += it->second;
	}
	
	double result = 0;
	if( method == UNKNOWN || method == Sc){

		// compute Hc
		const double Hc = H( classCounts, totalCount);

		//compute HT
		const double Ht = -( leftCount / totalCount * log2( leftCount / totalCount) 
					+  rightCount / totalCount * log2( rightCount / totalCount));

		const double HcLeft =  H( leftClassCounts, leftCount);
		const double HcRight = H( rightClassCounts, rightCount);

		// compute Ict
		const double Ict = Hc 
				 - leftCount / totalCount * HcLeft
				 - rightCount / totalCount * HcRight;

		result = (2 * Ict) / ( Hc + Ht);

	} else if( method == MutualInformation){

		const double Hc = H( classCounts, totalCount);
		const double HcLeft =  H( leftClassCounts, leftCount);
		const double HcRight = H( rightClassCounts, rightCount);

		// compute Ict
		const double Ict = Hc 
				 - leftCount / totalCount * HcLeft
				 - rightCount / totalCount * HcRight;

		result = Ict;

	} else if( method == Entropy){

		const double HcLeftPart =  H( leftClassCounts, totalCount);
		const double HcRightPart = H( rightClassCounts, totalCount);

		result = HcLeftPart + HcRightPart;

	} else if( method == Balanced || method == Mean){
		result = 0;
	}

	return result;
}


void TInformaionComputer::getBestThreshold( std::vector< TProjectedPoint> &data, float &threshold, double &score) const
{

	sort( data.begin(), data.end(), TInformaionComputer::TProjectedPoint::cmp);

	map< int, double> classCounts, leftClassCounts, rightClassCounts;

	double totalCount = 0;
	for( int i = 0; i < (int) data.size(); i++){
		classCounts[ data[ i].label] += 1.0;
		totalCount += 1.0;
	}

	rightClassCounts = classCounts;

	double leftCount = 0;
	double rightCount = totalCount;

	
	// compute the initial values

	// compute Hc
	const double Hc = H( classCounts, totalCount);

	//compute HT
	double Ht = 0;//-( leftCount / totalCount * log2( leftCount / totalCount) 
				  //+ rightCount / totalCount * log2( rightCount / totalCount));

	double HcLeft  = H( leftClassCounts, leftCount);
	double HcRight = H( rightClassCounts, rightCount);

	// compute Ict
	double Ict = 0;

	int bestThPos = -1;
	double bestScore = 0;

	if( method == Sc || method == MutualInformation){

		// try thresholds
		for( int i = 0; i < (int) data.size() - 1; i++){

			const int label = data[ i].label;

			// computing Ict as I(X;Y) = H(X) + H(Y) - H(X,Y)
			// this allows to compute it in one pass with minimal updates per sample
			// regardles the number of classes
			Ict -= Ht;
			if( leftClassCounts[ label] > 0.1){
				Ict += - leftClassCounts[ label] / totalCount * log2( leftClassCounts[ label] / totalCount);
			}
			Ict += - rightClassCounts[ label] / totalCount * log2( rightClassCounts[ label] / totalCount);

			leftCount++;
			leftClassCounts[ label]++;
			rightCount--;
			rightClassCounts[ label]--;

			// update Ht 
			if( leftCount != 0 && rightCount != 0){
				Ht = -( leftCount / totalCount * log2( leftCount / totalCount) 
					 + rightCount / totalCount * log2( rightCount / totalCount));
			} else {
				Ht = 0.0;
			}

			// there is now way that leftClassCounts[ label] could be 0 - there is the sample we are currently passing
			Ict += Ht;
			Ict -= - leftClassCounts[ label] / totalCount * log2( leftClassCounts[ label] / totalCount);
			if( rightClassCounts[ label] != 0){
				Ict -= - rightClassCounts[ label] / totalCount * log2( rightClassCounts[ label] / totalCount);
			}

			double currentScore = 0;
			
			if( method == Sc){
				currentScore = (2 * Ict) / ( Hc + Ht);
			} else if( method == MutualInformation){
				currentScore = Ict;
			}
		
			if( currentScore  > bestScore && data[ i].val != data[ i + 1].val ){
				bestThPos = i;
				bestScore = currentScore;
			}
		}


	} else if( method == Entropy){

		const double HcLeftPart  = H( leftClassCounts, totalCount);
		const double HcRightPart = H( rightClassCounts, totalCount);

		double entropy = HcLeftPart + HcRightPart;

		// try thresholds
		for( int i = 0; i < (int) data.size(); i++){

			const int label = data[ i].label;

			// computing entropy of p( T, C) - T for test, C for class
			// computed for all thresholds in one pass
			// first remove effect of the current sample and its class
			if( leftClassCounts[ label] > 0.1){
				entropy += leftClassCounts[ label] / totalCount * log2( leftClassCounts[ label] / totalCount);
			}
			entropy += rightClassCounts[ label] / totalCount * log2( rightClassCounts[ label] / totalCount);

			leftClassCounts[ label]++;
			rightClassCounts[ label]--;

			// there is now way that leftClassCounts[ label] could be 0 - there is the sample we are currently passing
			entropy -= leftClassCounts[ label] / totalCount * log2( leftClassCounts[ label] / totalCount);
			if( rightClassCounts[ label] != 0){
				entropy -= rightClassCounts[ label] / totalCount * log2( rightClassCounts[ label] / totalCount);
			}

			const double currentScore = entropy;
			
			if( currentScore  > bestScore){
				bestThPos = i;
				bestScore = currentScore;
			}
		}


	} else if( method == Balanced || method == Mean){

		double x = 0;
		double x2 = 0;

		for( int i = 0; i < (int) data.size(); i++){

			x  += data[ i].val;
			x2 += data[ i].val * data[ i].val;
		}

		x /= data.size();

		if( method == Balanced){
			threshold = (data[ data.size() / 2].val + data[ data.size() / 2 + 1].val) / 2.0;
		} else {
			threshold = x;
		}

			 
		score = x2 / data.size() - x * x;

		return;

	} else if( method == ScWithError){

		// try thresholds
		for( int i = 0; i < (int) data.size(); i++){

			const int label = data[ i].label;

			// This is absolutely nonsence, but it provides good results
			if( leftClassCounts[ label] > 0.1){
				Ict += - leftCount / totalCount * leftClassCounts[ label] / leftCount * log2( leftClassCounts[ label] / leftCount);
			}
			Ict += - rightCount / totalCount * rightClassCounts[ label] / rightCount * log2( rightClassCounts[ label] / rightCount);

			leftCount++;
			leftClassCounts[ label]++;
			rightCount--;
			rightClassCounts[ label]--;

			// update Ht 
			if( leftCount != 0 && rightCount != 0){
				Ht = -( leftCount / totalCount * log2( leftCount / totalCount) 
					 + rightCount / totalCount * log2( rightCount / totalCount));
			} else {
				Ht = 0.0;
			}

			// there is now way that leftClassCounts[ label] could be 0 - there is the sample we are currently passing
			Ict -= - leftCount / totalCount * leftClassCounts[ label] / leftCount * log2( leftClassCounts[ label] / leftCount);
			if( rightClassCounts[ label] != 0){
				Ict -= - rightCount / totalCount * rightClassCounts[ label] / rightCount * log2( rightClassCounts[ label] / rightCount);
			}

			const double currentScore = (2 * Ict) / ( Hc + Ht);
		
			if( currentScore  > bestScore){
				bestThPos = i;
				bestScore = currentScore;
			}
		}

	}

	score = bestScore;
	if( bestThPos >= 0){
		if( bestThPos + 1 < (int) data.size()){
			threshold = (data[ bestThPos].val + data[ bestThPos + 1].val) / 2.0;
		} else {
			threshold = data[ bestThPos].val;
		}
	}
}


