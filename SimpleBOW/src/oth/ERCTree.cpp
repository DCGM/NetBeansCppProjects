#include "ERCTree.h"

#include <cmath>
#include "opencv/cxcore.h"
#include <map>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

static int tempCount = 0;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TCMPTreeNode
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

TCMPTreeNode::TCMPTreeNode( const vector< float *>& data, const int dimension, const vector< int>& labels, const int Tmax, const TCodeBookBinaryTreeNode *_parrent, const int levelsLeft, const TInformaionComputer method)
	: TCodeBookBinaryTreeNode( _parrent)
{

	assert( data.size() == labels.size());

	if( levelsLeft == 0 || data.size() < 2){
		return;
	}

	int iterationCount = 0;

	CvRNG rng = cvRNG( rand());

	int bestDim1 = 0;
	int bestDim2 = 0;
	double bestSc = 0.0;

	for( int i = 0; i < Tmax; i++){

		// get the random dimension and threshold (it has unifirm distribution according the data
		const int dim1 = rand() % dimension;
		const int dim2 = rand() % dimension;

		double Sc = 0.0;

		vector< TInformaionComputer::TProjectedPoint> projData( data.size());
		for( int j = 0; j < (int) data.size(); j++){
			projData[ j].val = data[ j][ dim1] > data[ j][ dim2];
			projData[ j].label = labels[ j];
		}

		Sc = method.evaluateThreshold( projData, 0.5);

		if( Sc > bestSc){
			bestSc = Sc;
			bestDim1 = dim1;
			bestDim2 = dim2;
		}
	} 

	decDim1 = bestDim1;
	decDim2 = bestDim2;

	// split the data according to the best test
	vector< float *> leftData, rightData;
	vector< int> leftLabels, rightLabels;

	for( int i = 0; i < (int) data.size(); i++){
		if( data[ i][ decDim1] <= data[ i][ decDim2]){
			leftData.push_back( data[ i]);
			leftLabels.push_back( labels[ i]);
		} else {
			rightData.push_back( data[ i]);
			rightLabels.push_back( labels[ i]);
		}
	}

	// create the child nodes
	left = new TCMPTreeNode( leftData, dimension, leftLabels, Tmax, this, levelsLeft - 1, method);
	right = new TCMPTreeNode( rightData, dimension, rightLabels, Tmax, this, levelsLeft - 1, method);

	if( !left->OK()){
		delete left;
		left = NULL;
	}
	if( !right->OK()){
		delete right;
		right = NULL;
	}

	status = true;
}

TCMPTreeNode::TCMPTreeNode( std::istream &stream, const TCodeBookBinaryTreeNode *_parrent)
	: TCodeBookBinaryTreeNode( stream, _parrent)
{

	stream >> decDim1;
	stream >> decDim2;

	if( stream.fail()){
		throw( "Read error");
	}

	if( left != NULL){
		left = new TCMPTreeNode( stream, this);
	}
	if( right != NULL){
		right = new TCMPTreeNode( stream, this);
	}

	status = true;
}


void TCMPTreeNode::write( std::ostream &stream, const bool bin) const
{
	TCodeBookBinaryTreeNode::write( stream);
	stream << decDim1 << ' ';
	stream << decDim2 << ' ';

	if( stream.fail()){
		throw string( "Write error");
	}
	if( left != NULL){
		left->write( stream);
	}
	if( right != NULL){
		right->write( stream);
	}
}

bool TCMPTreeNode::goLeft( const float *point) const
{
	return point[ decDim1] <= point[ decDim2];
}


const string TCMPTreeNode::filePrefix = "CmpClusteringForestFile";

bool TCMPTreeNode::read( const std::string fileName, std::vector< TCMPTreeNode *> &trees)
{
	trees.clear();

	ifstream stream( fileName.data());
	
	if( !stream.good()){
		cerr << "Error: Unable to open file \"" << fileName << "\" for reading." << endl;
		return false;
	}

	string prefix = "";
	int treeCount = 0;
	stream >> prefix >> treeCount;

	if( prefix != filePrefix){
		// wrong file format
		return false;
	}

	for( int i = 0; i < treeCount && stream.good(); i++){
		trees.push_back( new TCMPTreeNode( stream, NULL));
		if( !trees.back()->OK()){
			cerr << "Error: Failed to load ERC tree from file \"" << fileName << "\" number " << i + 1 << "." << endl;
			delete trees.back();
			trees.pop_back();
			return false;
		}
	}

	if( trees.size() != treeCount){
		cerr << "Error: Only " << trees.size() << " trees were loaded from file \"" << fileName << "\", but the file should contain " << treeCount << " trees." << endl;
		return false;
	}

	return true;
}

bool TCMPTreeNode::write( const std::string fileName, std::vector< TCMPTreeNode *> &trees)
{
	ofstream stream( fileName.data());
	
	if( stream.fail()){
		cerr << "Error: Unable to open file \"" << fileName << "\" for writing." << endl;
		return false;
	}

	stream << filePrefix << " " << trees.size() << " ";

	for( int i = 0; i < (int) trees.size(); i++){
		trees[ i]->write( stream);
	}

	return true;
}

















inline float dotProduct( const vector< float> &data1, const float *data2, const int dim ){
	float sum = 0.0f;

	for( int i = 0; i < dim; i++){

		sum += data1[ i] * data2[ i];
	}

	return sum;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TPCTreeNode
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void getRandomThreshold( const vector< float *> &data, const int dim, const vector< int> &labels, const vector< float> &projection, float &threshold, double &score, const TInformaionComputer method)
{
	// compute the projection
	vector< TInformaionComputer::TProjectedPoint> projData( data.size());

	for( int i = 0; i < (int) data.size(); i++){
		projData[ i].label = labels[ i];
		projData[ i].val = dotProduct( projection, data[ i], dim);
	}

	threshold = projData[ rand() % projData.size()].val;
	
	score = method.evaluateThreshold( projData, threshold);
}



void getBestThreshold( const vector< float *> &data, const int dim, const vector< int> &labels, const vector< float> &projection, float &threshold, double &score, const TInformaionComputer method)
{
	// compute the projection

	vector< TInformaionComputer::TProjectedPoint> projData( data.size());

	for( int i = 0; i < (int) data.size(); i++){
		projData[ i].label = labels[ i];
		projData[ i].val = dotProduct( projection, data[ i], dim);
	}

	method.getBestThreshold( projData, threshold, score);

	/*tempCount++;
	double realSc = method.evaluateThreshold( projData, threshold);
	if(  (realSc - score) * (realSc - score) > (0.0001 * realSc) * (0.0001 * realSc) && (score > 0.0000001 || realSc > 0.0000001)){
		cout << "Sc difference count realSc Sc: " << tempCount << ' ' <<  realSc << ' ' << score << endl;
	}*/
}


void chooseProjection( CvMat *eigenValues, CvMat *eigenVectors, const int maxProjection, vector< float> & bestProjection)
{
	// get rid of negative eigen values
	cvMaxS( eigenValues, 0, eigenValues);
	cvPow( eigenValues, eigenValues, 2.0);

	// get accumulated eigen values
	vector< float> accEigenVal( eigenValues->cols, 0.0);
	accEigenVal[ 0] = (float) cvGetReal1D( eigenValues, 0);

	for( int i = 1; i < eigenValues->cols; i++){
		accEigenVal[ i] = accEigenVal[ i - 1] + (float) cvGetReal1D( eigenValues, i);
	}

	// get random value from 0 to sum of eigen values 
	float randThr = rand() / (float) RAND_MAX * accEigenVal[ min( maxProjection, (int) accEigenVal.size() - 1)];

	// get the random eigen vector ID
	int eigenVectorID = 0;
	while( accEigenVal[ eigenVectorID] < randThr){
		eigenVectorID++;
	}

	// get the random projection
	for( int j = 0; j < eigenVectors->cols; j++){
		bestProjection[ j] = (float) cvGetReal2D( eigenVectors, eigenVectorID, j);
	}

	cout << "EV id " << eigenVectorID << " EV sum fraction " << (float) cvGetReal1D( eigenValues, eigenVectorID) / accEigenVal.back() << endl;
}	

std::vector<float> TPCTreeNode::normalDistribution;

float TPCTreeNode::getNormalDistrSideIntegral( const float point) const
{
	static const int normDistSize = 500;
	static const float maxPoint = 5;

	if( normalDistribution.empty()){

		normalDistribution = vector< float>( normDistSize, 0); 

		const double normFactor = 1 / 2.506628274;
		for( int i = 0; i < normDistSize; i++){
			const double o = i / (double) (normalDistribution.size() - 1) * maxPoint;
			normalDistribution[ i] = normFactor * exp( - o * o / 2) * ( maxPoint / (normDistSize - 0.5));
		}

		normalDistribution[ 0] /= 2;

		for( int i = normalDistribution.size() - 1; i > 0; i--){
			normalDistribution[ i - 1] += normalDistribution[ i];
		}
	}

	const int index = (int) (fabsf( point) / maxPoint * ( normDistSize - 1) + 0.5);

	return (index < normDistSize)? (normalDistribution[ index]) : (0.0f);
}



TPCTreeNode::TPCTreeNode( const vector< float *>& allData, const int dimension, vector< int>& allLabels,
	const double Smin, const int Tmax, const TCodeBookBinaryTreeNode *_parrent, const int levelsLeft, 
	const bool randomThreshold, const TInformaionComputer &method, const string & projectionSelectionMethod, 
	const int samplesToEstimateCovariance)
	: TCodeBookBinaryTreeNode( _parrent), standardDeviation( 1.0f)
{

	assert( allData.size() == allLabels.size());

	if( levelsLeft == 0 || allData.size() < 2){
		return;
	}


	vector< float *> sampledData;
	vector< int> sampledLabels;
	const int classesToUse = (int) Smin;

	if( classesToUse > 1){
		
		map< int, int> sortedLabels;
		for( int i = 0; i < (int) allLabels.size(); i++){
			sortedLabels[ allLabels[ i]]++;
		}

		vector< int> labels;
		for( map< int, int>::const_iterator it = sortedLabels.begin(); it != sortedLabels.end(); it++){
			labels.push_back( it->first);
		}

		while( (int) labels.size() > classesToUse){
			labels[ rand() % labels.size()] = labels.back();
			labels.pop_back();
		}

		sortedLabels.clear();
		for( int i = 0; i < (int) labels.size(); i++){
			sortedLabels[ labels[ i]] = 1;
		}

		for( int i = 0; i < (int) allData.size(); i++){
			if( sortedLabels.find( allLabels[ i]) != sortedLabels.end()){
				sampledData.push_back( allData[ i]);
				sampledLabels.push_back( allLabels[ i]);
			}
		}
	}

	const vector< float *> &data = (classesToUse > 0)?( sampledData):( allData);
	const vector< int> &labels = (classesToUse > 0)?( sampledLabels):( allLabels);



	int iterationCount = 0;

	CvRNG rng = cvRNG( rand());

	float bestThreshold = +1e20f;
	vector< float> bestProjection( dimension, 0.0f);
	double bestSc = 0.0;

	if( projectionSelectionMethod.empty() || projectionSelectionMethod == "RND" ){

		for( int i = 0; i < Tmax; i++){

			// get the random projection
			vector< float> tempProjection( dimension, 0.0);

			float randVal = 0.0f;
			CvMat rndMat = cvMat( 1, 1, CV_32F, &randVal);
			for( int j = 0; j < dimension; j++){
				cvRandArr( &rng, &rndMat, CV_RAND_NORMAL, cvScalarAll( 0), cvScalarAll( 1));
				tempProjection[ j] = randVal;
			}
			
			float tempThreshold = 0;
			double tempScore = 0;
			

			if( randomThreshold){
				getRandomThreshold( data, dimension, labels, tempProjection, tempThreshold, tempScore, method);
			} else {
				getBestThreshold( data, dimension, labels, tempProjection, tempThreshold, tempScore, method);
			}

			if( tempScore > bestSc){
				bestSc = tempScore;
				bestProjection = tempProjection;
				bestThreshold = tempThreshold;
			}
		} 
		cout << "RANDOM " << bestSc;

	} else if( projectionSelectionMethod == "PCA" ) {
		
		vector< int> samplesLeft( labels.size());
		for( int i = 0; i < (int) samplesLeft.size(); i++) {
			samplesLeft[ i] = i;
		}

		// prepare data for PCA
		CvMat * samplesForCovariance = cvCreateMat( min( samplesToEstimateCovariance, (int)data.size()), dimension, CV_32F);

		for( int i = 0; i < samplesForCovariance->rows; i++){

			// select and remove random sample
			const int sampleID = rand() % samplesLeft.size();
			const float *sample( data[ samplesLeft[ sampleID]]);
			samplesLeft[ sampleID] = samplesLeft.back();
			samplesLeft.pop_back();

			// copy the data of the selected sample
			float *dataPtr = (float *)cvPtr2D( samplesForCovariance, i, 0);
			for( int j = 0; j < samplesForCovariance->cols; j++){
				*(dataPtr++) = sample[ j];
			}
		}


		// Get Covariance matrix
		CvMat* covMatrix = cvCreateMat( samplesForCovariance->cols, samplesForCovariance->cols, CV_32F);
		CvMat* avg = cvCreateMat( 1, samplesForCovariance->cols, CV_32F);
		cvSet( avg, cvScalarAll( 0.0));

		cvCalcCovarMatrix( (const CvArr **) &samplesForCovariance, 1, covMatrix, avg,CV_COVAR_ROWS | CV_COVAR_NORMAL);
		cvReleaseMat( &samplesForCovariance);
		cvReleaseMat( &avg);

		// compute eigen vectors
		CvMat* eigenValues = cvCreateMat( 1, covMatrix->cols, CV_32F);
		CvMat* eigenVectorsMat = cvCreateMat( covMatrix->cols, covMatrix->cols, CV_32F);

		cvSVD( covMatrix, eigenValues, eigenVectorsMat, NULL, CV_SVD_U_T | CV_SVD_MODIFY_A);
		cvReleaseMat( &covMatrix);

		chooseProjection( eigenValues, eigenVectorsMat, Tmax, bestProjection);

		if( randomThreshold){
			for( int i = 0; i < Tmax; i++){

				float tempThreshold = 0;
				double tempScore = 0;

				getRandomThreshold( data, dimension, labels, bestProjection, tempThreshold, tempScore, method);

				if( tempScore > bestSc){
					bestSc = tempScore;
					bestThreshold = tempThreshold;
				}
			}
		} else {
			getBestThreshold( data, dimension, labels, bestProjection, bestThreshold, bestSc, method);
		}

		cvReleaseMat( &eigenValues);
		cvReleaseMat( &eigenVectorsMat);

	} else if( projectionSelectionMethod == "LDP"){

		// sort samples according to their label
		map< int, vector< int> > sortedLabel;
		for( int i = 0; i < (int) labels.size(); i++){
			sortedLabel[ labels[ i]].push_back( i);
		}

		// compute average class sample count
		cout << "Samples per class: " << labels.size() / (double) sortedLabel.size() << endl;


		CvMat * eigenValues = cvCreateMat( 1, dimension, CV_32F);
		CvMat * eigenVectorsMat = cvCreateMat( dimension, dimension, CV_32F);

		LDP( data, dimension, labels, eigenVectorsMat, eigenValues, samplesToEstimateCovariance, false, true, 0);//(int) Smin);

		if( cvGetReal1D( eigenValues, 0) > 1e-5){

			chooseProjection( eigenValues, eigenVectorsMat, Tmax, bestProjection);

			if( cvGetReal1D( eigenValues, 0) > 1e-20 && randomThreshold){
				for( int i = 0; i < Tmax; i++){

					float tempThreshold = 0;
					double tempScore = 0;

					getRandomThreshold( data, dimension, labels, bestProjection, tempThreshold, tempScore, method);

					if( tempScore > bestSc){
						bestSc = tempScore;
						bestThreshold = tempThreshold;
					}
				}
			} else {
				getBestThreshold( data, dimension, labels, bestProjection, bestThreshold, bestSc, method);
			}
			
		}
		
		cvReleaseMat( &eigenValues);
		cvReleaseMat( &eigenVectorsMat);

	} else {
		cerr << "ERROR: Unknown projectionSelectionMethod \"" << projectionSelectionMethod << "\" when building PC-TREE. Possible values are RND, PCA, LDP, LDP-SVD." << endl;
		exit( -1);
	}

	if( bestSc > 1e-6){

		projection = bestProjection;
		threshold = bestThreshold;

		// split the data according to the best test
		vector< float *> leftData, rightData;
		vector< int> leftLabels, rightLabels;

		map< int , pair< int, pair< double, double> > > mapXX2;


		for( int i = 0; i < (int) data.size(); i++){
			
			const float projectedData = dotProduct( projection, data[ i], dimension);
				
			mapXX2[ labels[ i] ].first++;
			mapXX2[ labels[ i] ].second.first += projectedData;
			mapXX2[ labels[ i] ].second.second += projectedData * projectedData;

			if(  projectedData <= threshold){
				leftData.push_back( data[ i]);
				leftLabels.push_back( labels[ i]);
			} else {
				rightData.push_back( data[ i]);
				rightLabels.push_back( labels[ i]);
			}
		}

		double x = 0;
		double x2 = 0;

		standardDeviation = 0;
		for( map< int , pair< int, pair< double, double> > >::const_iterator it = mapXX2.begin(); it != mapXX2.end(); it++){
			standardDeviation += sqrt( max( 1e-10, it->second.second.second / it->second.first - pow( (double)(it->second.second.first / it->second.first), 2)));
			x += it->second.second.first;
			x2 += it->second.second.second;
		}
		standardDeviation /= mapXX2.size();
	
		cout << "Data StdDev: " << sqrt( x2 / data.size() - pow( x / data.size(), 2)) << " Class StdDev: " << standardDeviation << endl;


		cout << levelsLeft << " Sc: " << bestSc << " Thr: " << threshold << ' ' << " Left: " << leftData.size() << " Right: " << rightData.size() << endl;

		// create the child nodes
		left = new TPCTreeNode( leftData, dimension, leftLabels, 0/*Smin,*/, Tmax, this, levelsLeft - 1, randomThreshold, method, projectionSelectionMethod);
		right = new TPCTreeNode( rightData, dimension, rightLabels, 0/*Smin*/, Tmax, this, levelsLeft - 1, randomThreshold, method, projectionSelectionMethod);

		if( !left->OK()){
			delete left;
			left = NULL;
		}
		if( !right->OK()){
			delete right;
			right = NULL;
		}
	}

	cout << "RETURN " << endl;

	status = true;
}


TPCTreeNode::TPCTreeNode( std::istream &stream, const TCodeBookBinaryTreeNode *_parrent, const bool binary)
	: TCodeBookBinaryTreeNode( stream, _parrent, binary)
{
	if( binary){

		int dataDimension = 0;
		stream.read( (char *) &standardDeviation, sizeof( float));
		stream.read( (char *) &threshold, sizeof( float));
		stream.read( (char *) &dataDimension, sizeof( int));
	
		projection.resize( dataDimension);

		for( int i = 0; i < dataDimension; i++){
			stream.read( (char *) &projection[ i], sizeof( float));
		}

		if( stream.fail()){
			throw( "Read error");
		}

	} else {

		stream >> standardDeviation;
		stream >> threshold;

		int dataDimension = 0;
		stream >> dataDimension;
		projection.resize( dataDimension);

		for( int i = 0; i < dataDimension; i++){
			stream >> projection[ i];
		}

		if( stream.fail()){
			throw( "Read error");
		}
	}

	if( left != NULL){
		left = new TPCTreeNode( stream, this, binary);
	}
	if( right != NULL){
		right = new TPCTreeNode( stream, this, binary);
	}

	status = true;
}


void TPCTreeNode::write( std::ostream &stream, const bool bin) const
{
	TCodeBookBinaryTreeNode::write( stream, true);

	const int dim = projection.size();

	stream.write( (const char *) &standardDeviation, sizeof( float));
	stream.write( (const char *) &threshold, sizeof( float));
	stream.write( (const char *) &dim, sizeof( int));
	for( int i = 0; i < (int) projection.size(); i++){
		stream.write( (const char *) &projection[ i], sizeof( float));
	}

	if( stream.fail()){
		throw string( "Write error");
	}
	if( left != NULL){
		left->write( stream);
	}
	if( right != NULL){
		right->write( stream);
	}
}

bool TPCTreeNode::goLeft( const float *point) const
{
	return dotProduct( projection, point, projection.size()) <= threshold;
}

void TPCTreeNode::getWords( const float * point, std::vector< int>& words) const
{
	const float projectionVal = dotProduct( projection, point, projection.size());

	const float leftProbability = (projectionVal <= threshold) ? (1.0f - getNormalDistrSideIntegral( (projectionVal - threshold) / standardDeviation)): getNormalDistrSideIntegral( (projectionVal - threshold) / standardDeviation);

	int leftCount = 0;

	for( int i = 0; i < (int) words.size(); i++){
		if( rand() / (float) RAND_MAX <= leftProbability){
			leftCount++;
		}
	}


	if( leftCount > 0){
	
		if( left != NULL){
			vector< int> leftWords( leftCount);
			((TPCTreeNode *) left)->getWords( point, leftWords);

			for( int i = 0; i < leftCount; i++){
				words[ i] = leftWords[ i];
			}
		} else {
			for( int i = 0; i < leftCount; i++){
				words[ i] = leftID;
			}
		}
	}

	if( leftCount != words.size()){

		if( right != NULL){
			vector< int> rightWords( words.size() - leftCount);
			((TPCTreeNode *) right)->getWords( point, rightWords);

			for( int i = leftCount; i < (int) words.size(); i++){
				words[ i] = rightWords[ i - leftCount];
			}
		} else {
			for( int i = leftCount; i < (int) words.size(); i++){
				words[ i] = rightID;
			}
		}
	}


}




const string TPCTreeNode::filePrefix = "ProjectionClusteringForestFile";
const string TPCTreeNode::filePrefixBin = "ProjectionClusteringForestFileBinary";

bool TPCTreeNode::read( const std::string fileName, std::vector< TPCTreeNode *> &trees)
{
	trees.clear();

	ifstream stream( fileName.data());
	
	if( !stream.good()){
		cerr << "Error: Unable to open file \"" << fileName << "\" for reading." << endl;
		return false;
	}

	string prefix = "";
	int treeCount = 0;
	stream >> prefix >> treeCount;

	bool binary = false;

	if( prefix == filePrefixBin){
		stream.close();

		stream.open(fileName.data(), ios_base::binary);
		
		if( !stream.good()){
			cerr << "Error: Unable to open file \"" << fileName << "\" for reading." << endl;
			return false;
		}

		stream >> prefix >> treeCount;

		stream.ignore( 1);

		binary = true;
	} else if( prefix != filePrefix){
		return false;
	}


	for( int i = 0; i < treeCount && stream.good(); i++){
		trees.push_back( new TPCTreeNode( stream, NULL, binary));
		if( !trees.back()->OK()){
			cerr << "Error: Failed to load PC tree from file \"" << fileName << "\" number " << i + 1 << "." << endl;
			delete trees.back();
			trees.pop_back();
			return false;
		}
	}

	if( trees.size() != treeCount){
		cerr << "Error: Only " << trees.size() << " trees were loaded from file \"" << fileName << "\", but the file should contain " << treeCount << " trees." << endl;
		return false;
	}

	return true;
}

bool TPCTreeNode::write( const std::string fileName, std::vector< TPCTreeNode *> &trees)
{
	ofstream stream( fileName.data(), ios_base::binary);
	
	if( stream.fail()){
		cerr << "Error: Unable to open file \"" << fileName << "\" for writing." << endl;
		return false;
	}

	stream << filePrefixBin << " " << trees.size() << " ";

	for( int i = 0; i < (int) trees.size(); i++){
		trees[ i]->write( stream);
	}

	return true;
}
















//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TERCTreeNode
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

TERCTreeNode::TERCTreeNode( const vector< float *>& data, const int dimension, const vector< int>& labels, const double Smin, const int Tmax, const TCodeBookBinaryTreeNode *_parrent, const int levelsLeft, const bool randomThreshold, const TInformaionComputer method)
	: TCodeBookBinaryTreeNode( _parrent)
{

	assert( data.size() == labels.size());

	if( levelsLeft == 0 || data.size() < 2){
		return;
	}


	int iterationCount = 0;

	// compute mean an variance for generating thresholds later
	vector< float> x( dimension, 0.0f);
	vector< float> x2( dimension, 0.0f);


	for( int i = 0; i < (int)data.size(); i++){
		for( int j = 0; j < dimension; j++){
			x[ j] += data[i][ j];
			x2[ j] += data[i][ j] * data[i][ j];
		}
	}

	for( int j = 0; j < dimension; j++){
		x[ j] /= (float) data.size();
		x2[ j] = (float) sqrtf( x2[ j] / (float)data.size() - x[ j] * x[ j]);
	}

	CvRNG rng = cvRNG( rand());

	float bestThreshold = -1e20f;
	int bestDimension = 0;
	double bestSc = 0.0;


	for( int i = 0; i < Tmax && bestSc < Smin; i++){

		// get the random dimension and threshold (it has unifirm distribution according the data
		const int dim = rand() % dimension;
		float threshold = 0.0f;
		double Sc = 0.0;

		vector< TInformaionComputer::TProjectedPoint> projData( data.size());
		for( int j = 0; j < (int) data.size(); j++){
			projData[ j].val = data[ j][ dim];
			projData[ j].label = labels[ j];
		}

		if( randomThreshold){

			CvMat rndMat = cvMat( 1, 1, CV_32F, &threshold);
			cvRandArr( &rng, &rndMat, CV_RAND_NORMAL, cvScalarAll( x[ dim]), cvScalarAll( x2[ dim]));

			Sc = method.evaluateThreshold( projData, threshold);

		} else {
			method.getBestThreshold( projData, threshold, Sc);

/*			tempCount++;
			double realSc = method.evaluateThreshold( projData, threshold);
			if(  (realSc - Sc) * (realSc - Sc) > (0.0001 * realSc) * (0.0001 * realSc) && (Sc > 0.0000001 || realSc > 0.0000001)){
				cout << "Sc difference count realSc Sc: " << tempCount << ' ' <<  realSc << ' ' << Sc << endl;
			}*/

		}

		if( Sc > bestSc){
			bestSc = Sc;
			bestDimension = dim;
			bestThreshold = threshold;
		}
	} 

	if( bestSc > 1e-6){

		decisionDimension = bestDimension;
		decisionThreshold = bestThreshold;

		// split the data according to the best test
		vector< float *> leftData, rightData;
		vector< int> leftLabels, rightLabels;

		for( int i = 0; i < (int) data.size(); i++){
			if( data[ i][ decisionDimension] <= decisionThreshold){
				leftData.push_back( data[ i]);
				leftLabels.push_back( labels[ i]);
			} else {
				rightData.push_back( data[ i]);
				rightLabels.push_back( labels[ i]);
			}
		}

		// create the child nodes
		left = new TERCTreeNode( leftData, dimension, leftLabels, Smin, Tmax, this, levelsLeft - 1, randomThreshold, method);
		right = new TERCTreeNode( rightData, dimension, rightLabels, Smin, Tmax, this, levelsLeft - 1, randomThreshold, method);

		if( !left->OK()){
			delete left;
			left = NULL;
		}
		if( !right->OK()){
			delete right;
			right = NULL;
		}
	}

	status = true;
}

TERCTreeNode::TERCTreeNode( std::istream &stream, const TCodeBookBinaryTreeNode *_parrent)
	: TCodeBookBinaryTreeNode( stream, _parrent)
{

	stream >> decisionDimension;
	stream >> decisionThreshold;

	if( stream.fail()){
		throw( "Read error");
	}

	if( left != NULL){
		left = new TERCTreeNode( stream, this);
	}
	if( right != NULL){
		right = new TERCTreeNode( stream, this);
	}

	status = true;
}


void TERCTreeNode::write( std::ostream &stream, const bool bin) const
{
	TCodeBookBinaryTreeNode::write( stream);
	stream << decisionDimension << ' ';
	stream << decisionThreshold << ' ';

	if( stream.fail()){
		throw string( "Write error");
	}
	if( left != NULL){
		left->write( stream);
	}
	if( right != NULL){
		right->write( stream);
	}
}

bool TERCTreeNode::goLeft( const float *point) const
{
	return point[ decisionDimension] <= decisionThreshold;
}

const string TERCTreeNode::filePrefix = "ExtreamlyRandomizedClusteringForestFile";

bool TERCTreeNode::read( const std::string fileName, std::vector< TERCTreeNode *> &trees)
{
	trees.clear();

	ifstream stream( fileName.data());
	
	if( !stream.good()){
		cerr << "Error: Unable to open file \"" << fileName << "\" for reading." << endl;
		return false;
	}

	string prefix = "";
	int treeCount = 0;
	stream >> prefix >> treeCount;

	if( prefix != filePrefix){
		// wrong file format
		return false;
	}

	for( int i = 0; i < treeCount && stream.good(); i++){
		trees.push_back( new TERCTreeNode( stream, NULL));
		if( !trees.back()->OK()){
			cerr << "Error: Failed to load ERC tree from file \"" << fileName << "\" number " << i + 1 << "." << endl;
			delete trees.back();
			trees.pop_back();
			return false;
		}
	}

	if( trees.size() != treeCount){
		cerr << "Error: Only " << trees.size() << " trees were loaded from file \"" << fileName << "\", but the file should contain " << treeCount << " trees." << endl;
		return false;
	}

	return true;
}

bool TERCTreeNode::write( const std::string fileName, std::vector< TERCTreeNode *> &trees)
{
	ofstream stream( fileName.data());
	
	if( stream.fail()){
		cerr << "Error: Unable to open file \"" << fileName << "\" for writing." << endl;
		return false;
	}

	stream << filePrefix << " " << trees.size() << " ";

	for( int i = 0; i < (int) trees.size(); i++){
		trees[ i]->write( stream);
	}

	return true;
}
