#include "ProjKMeans.h"

#include <cmath>
#include <map>
#include <fstream>
#include <ctime>
#include <algorithm>

#include "InfMethods.h"

#include "opencv/highgui.h"

using namespace std;

const std::string TProjKmeans::firstLineString = "Projection+K-means";

const int TProjKmeans::samplesToEstimateCovariance = 15000;
const bool TProjKmeans::normDifferences = false;
const bool TProjKmeans::myCovariance = true;

void sortEVV( CvMat * eVal, CvMat *eVec){
/*	
	CvMat *temp = cvCreateMat( 1, eVec->cols, eVec->type);
	CvMat * row1 = cvCreateMatHeader( 1, eVec->cols, eVec->type);
	CvMat * row2 = cvCreateMatHeader( 1, eVec->cols, eVec->type);

	for( int i = 0; i < eVal->cols; i++){
	
		// find best
		float bestVal = -1e10;
		int bestID = -1;
		for( int j = i; j < eVal->cols; j++){
			if( eVal->data.db[ j] > bestVal){
				bestVal = eVal->data.db[ j];
				bestID = j;
			}
		}

		// switch eVal
		const double d = eVal->data.db[ i];
		eVal->data.db[ i] = eVal->data.db[ bestID];
		eVal->data.db[ bestID] = d;

		cvGetRow( eVec, row1, i);
		cvGetRow( eVec, row2, bestID);

		cvCopy( row1, temp);
		cvCopy( row2, row1);
		cvCopy( temp, row2);
	}

	cvReleaseMat( &temp);
	cvReleaseMat( &row1);
	cvReleaseMat( &row2);*/
}


TProjKmeans::TProjKmeans( std::string filename)
	: statusOK( false), projection( NULL), centers( NULL), projectedPoint( NULL), tree( NULL), avgWordDistance( 0.0), exactKNN( false), inputVectorDimension( 0)
{
	ifstream file( filename.data());
	if( !file.good()){
		cerr << "Error: Unable to open file \"" << filename << "\" for reading in TProjKmeans." << endl;
		return;
	}

	string str;
	file >> str;

	if( str != firstLineString){
		if( file.fail()){
			cerr << "Error: Unable to read from file \"" << filename << "\" in TProjKmeans." << endl;
		}
		return;
	}

	file >> exactKNN;

	int rows = 0;
	file >> rows;
	if( file.fail()){
		cerr << "Error while reading from file \"" << filename << "\" in TProjKmeans." << endl;
		exit( -1);
	}

	int cols = 0;
	file >> cols;
	if( file.fail()){
		cerr << "Error while reading from file \"" << filename << "\" in TProjKmeans." << endl;
		exit( -1);
	}
	inputVectorDimension = cols;


	// read the projections
	if( rows != 0 && cols != 0){
		projection = cvCreateMat( rows, cols, CV_32F);

		for( int r = 0; r < projection->rows; r++){
			for( int c = 0; c < projection->cols; c++){
				float f;
				file >> f;
				if( file.fail()){
					cerr << "Error while reading from file \"" << filename << "\" in TProjKmeans." << endl;
					exit( -1);
				}
				cvSetReal2D( projection, r, c, f);
			}
		}
	}

	file >> rows;
	if( file.fail()){
		cerr << "Error while reading from file \"" << filename << "\" in TProjKmeans." << endl;
		exit( -1);
	}

	file >> cols;
	if( file.fail()){
		cerr << "Error while reading from file \"" << filename << "\" in TProjKmeans." << endl;
		exit( -1);
	}
	if( inputVectorDimension == 0){
		inputVectorDimension = cols;
	}

	centers = cvCreateMat( rows, cols, CV_32F);

	// read cluster centers
	for( int r = 0; r < centers->rows; r++){
		for( int c = 0; c < centers->cols; c++){
			float f;
			file >> f;
			if( file.fail()){
				cerr << "Error while reading from file \"" << filename << "\" in TProjKmeans." << endl;
				exit( -1);
			}
			cvSetReal2D( centers, r, c, f);
		}
	}


	// get avgWordDistance if it is present in the file
	string distanceStr;
	file >> distanceStr;
	if( distanceStr == "avgWordDistance"){
		file >> avgWordDistance;
	}

	// create a kd-tree for approximate nearest neighbor searches
	tree = cvCreateKDTree( centers);

	if( projection != NULL){
		projectedPoint = cvCreateMat( 1, projection->rows, projection->type); 
	}

    statusOK = true;
}

void TProjKmeans::write( std::string filename)
{
	ofstream file( filename.data());
	if( !file.good()){
		cerr << "Error: Unable to open file \"" << filename << "\" for writing in TProjKmeans." << endl;
		return;
	}

	file << firstLineString << endl;

	file << exactKNN << endl;

	// write projections
	if( projection == NULL){
		file << "0 0" << endl << endl;
	} else {
		file << projection->rows << " " << projection->cols << endl;

		for( int r = 0; r < projection->rows; r++){
			for( int c = 0; c < projection->cols; c++){
				file << cvGetReal2D( projection, r, c) << ' ';
			}
			file << endl;
		}
		file << endl;
	}


	// write cluster centers
	file << centers->rows << " " << centers->cols << endl;
	for( int r = 0; r < centers->rows; r++){
		for( int c = 0; c < centers->cols; c++){
			file << cvGetReal2D( centers, r, c) << ' ';
		}
		file << endl;
	}


	// write avgWordDistance
	file << "avgWordDistance " << getAvgWordDistance() << endl;

	if( file.fail()){
		cerr << "Error: While writing to file \"" << filename << "\" in TProjKmeans." << endl;
		return;
	}

}

/*static inline int findExactNN1( CvMat *point, CvMat *centers){

	int bestID = -1;
	double bestDist = 1e30;
	
	for( int i = 0; i < centers->rows; i++){

		const double distance = L2Distance( point->data.fl, (const float *) cvPtr2D( centers, i, 0), point->cols);

		if( distance < bestDist){
			bestDist = distance;
			bestID = i;
		}
	}

	return bestID;
}*/


static inline int findExactNN( CvMat *point, CvMat *centers){

	int bestID = -1;
	double bestDist = 1e30;
	
	// search for the L2 nearest cluster
	for( int i = 0; i < centers->rows; i++){

		const double distance = L2Distance( point->data.fl, (const float *) cvPtr2D( centers, i, 0), centers->cols, bestDist);

		if( distance < bestDist){
			bestDist = distance;
			bestID = i;
		}
	}

	return bestID;
}

/** this is for soring value/ID pairs*/
struct tRes{
	double val;
	int id;
};

/** this if for somparing value/ID pairs during sorting*/
bool tResCmp( tRes &a, tRes &b){
	return a.val < b.val;
}



void findExactNN( CvMat *point, CvMat *centers, vector< int> &nn, vector< double> &bestDistances)
{
	tRes * results = new tRes[ centers->rows];

	// compute distances to all clusters
	for( int i = 0; i < centers->rows; i++){
		results[ i].val = L2Distance( point->data.fl, (const float *) cvPtr2D( centers, i, 0), centers->cols, bestDistances.back());
		results[ i].id = i;
	}

	// sort clusters accoring to their distances
	partial_sort( results, results + nn.size(), results + centers->rows, tResCmp);

	// copy the required number of nearest clusters
	for( int i = 0; i < (int) nn.size(); i++){
		nn[ i] = results[ i].id;
		bestDistances[ i] = results[ i].val;
	}

	delete results;
}

double TProjKmeans::getAvgWordDistance(const int estimateCount)
{
	// if the avg distace has been already computed, return the value
	if( avgWordDistance != 0){
		return avgWordDistance;
	}

	// prepare list of cluster IDs
	vector< int> clustersLeft;
	for( int i = 0; i < centers->rows; i++){
		clustersLeft.push_back( i);
	}

	double sum = 0;
	int count = 0;
	for( int i = 0; i < estimateCount && !clustersLeft.empty(); i++){

		// get random cluster which has not yet been used
		const int index = rand() % clustersLeft.size();
		const int id1 = clustersLeft[ index];

		clustersLeft[ index] = clustersLeft.back();
		clustersLeft.pop_back();

		// compute a distance to the closest cluster
		double bestDistance = 1e30;
		for( int id2 = 0; id2 < centers->rows; id2++){
			if( id1 != id2){
				bestDistance = min( bestDistance, L2Distance( (const float *) cvPtr2D( centers, id1, 0), (const float *) cvPtr2D( centers, id2, 0), centers->cols));
			}
		}

		sum += bestDistance;
		count++;
	}

	avgWordDistance = sum / count;

	// compute average closest distance and return it
	return sum / count;
}


int TProjKmeans::getWord( const float *point) const
{
	int nearestID;
	CvMat nearestMat = cvMat( 1, 1, CV_32S, &nearestID);
	double distance;
	CvMat distanceMat = cvMat( 1, 1, CV_64F, &distance);

	if( projection != NULL){

		CvMat sample = cvMat( 1, projection->cols, CV_32F, (void *) point);

		cvGEMM( &sample, projection, 1.0, NULL, 0.0, projectedPoint, CV_GEMM_B_T);

		if( exactKNN){
			nearestID = findExactNN( projectedPoint, centers);
		} else {
			cvFindFeatures( tree, projectedPoint, &nearestMat, &distanceMat, 1, 200);
		}

	} else {
		
		CvMat sample = cvMat( 1, centers->cols, CV_32F, (void *) point);

		if( exactKNN){
			nearestID = findExactNN( &sample, centers);
		} else {
			cvFindFeatures( tree, &sample, &nearestMat, &distanceMat, 1, 200);
		}
	}

	return nearestID;
}

void TProjKmeans::getWord( const float *point, vector< int> &nn) const
{
	CvMat nearestMat = cvMat( 1, nn.size(), CV_32S, &nn[ 0]);
	vector< double> distance( nn.size());
	CvMat distanceMat = cvMat( 1, nn.size(), CV_64F, &distance[0]);


	if( projection != NULL){

		CvMat sample = cvMat( 1, projection->cols, CV_32F, (void *) point);

		cvGEMM( &sample, projection, 1.0, NULL, 0.0, projectedPoint, CV_GEMM_B_T);

		if( exactKNN){
			findExactNN( projectedPoint, centers, nn, distance);
		} else {
			cvFindFeatures( tree, projectedPoint, &nearestMat, &distanceMat, nn.size(), max( 200, (int)nn.size()));
		}

	} else {
		CvMat sample = cvMat( 1, centers->cols, CV_32F, (void *) point);

		if( exactKNN){
			findExactNN( &sample, centers, nn, distance);
		} else {
			cvFindFeatures( tree, &sample, &nearestMat, &distanceMat, nn.size(), max( 200, (int)nn.size()));
		}
	}
}

void TProjKmeans::getWord( const float *point, vector< int> &nn, vector< double> &distance) const
{
	assert( nn.size() == distance.size());

	for( int i = 0; i < (int) nn.size(); i++){
		nn[ i] = -1;
		distance[ i] = 1e30;
	}

	CvMat nearestMat = cvMat( 1, nn.size(), CV_32S, &nn[ 0]);
	CvMat distanceMat = cvMat( 1, nn.size(), CV_64F, &distance[0]);

	if( projection != NULL){

		CvMat sample = cvMat( 1, projection->cols, CV_32F, (void *) point);

		cvGEMM( &sample, projection, 1.0, NULL, 0.0, projectedPoint, CV_GEMM_B_T);


		if( exactKNN){
			findExactNN( projectedPoint, centers, nn, distance);
		} else {
			cvFindFeatures( tree, projectedPoint, &nearestMat, &distanceMat, nn.size(), max( 200, (int)nn.size()));
		}

	} else {
		CvMat sample = cvMat( 1, centers->cols, CV_32F, (void *) point);

		if( exactKNN){
			findExactNN( &sample, centers, nn, distance);
		} else {
			cvFindFeatures( tree, &sample, &nearestMat, &distanceMat, nn.size(), max( 200, (int)nn.size()));
		}
	}
}


#define SHOW_DATA 0

void TProjKmeans::showData( CvMat *projectedData, const vector< int> &labels){
#if SHOW_DATA

	vector< CvScalar> colors;
	colors.push_back( cvScalar( 255, 0, 0));
	colors.push_back( cvScalar( 0, 255, 0));
	colors.push_back( cvScalar( 0, 0, 255));
	colors.push_back( cvScalar( 255, 255, 0));
	colors.push_back( cvScalar( 255, 0, 255));
	colors.push_back( cvScalar( 0, 255, 255));
	colors.push_back( cvScalar( 255, 255, 255));
	colors.push_back( cvScalar( 128, 0, 0));
	colors.push_back( cvScalar( 0, 128, 0));
	colors.push_back( cvScalar( 0, 0, 128));
	colors.push_back( cvScalar( 128, 128, 0));
	colors.push_back( cvScalar( 128, 0, 128));
	colors.push_back( cvScalar( 0, 128, 128));
	colors.push_back( cvScalar( 128, 128, 128));

	// show the clusters
	int second = 1;
	while( true){
		IplImage *img = cvCreateImage( cvSize( 800, 600), IPL_DEPTH_8U, 3);
		cvSet( img, cvScalar( 0));

		double minX = 1e100;
		double maxX = -1e100;
		double minY = 1e100;
		double maxY = -1e100;
		for( int i = 0; i < projectedData->rows; i++){
			minX = min( minX, cvGetReal2D( projectedData, i, 0));
			maxX = max( maxX, cvGetReal2D( projectedData, i, 0));
			minY = min( minY, cvGetReal2D( projectedData, i, second));
			maxY = max( maxY, cvGetReal2D( projectedData, i, second));
		}

		for( int i = 0; i < projectedData->rows; i++){
			const double x = cvGetReal2D( projectedData, i, 0);
			const double y = cvGetReal2D( projectedData, i, second);
			cvSet2D( img, (int)(( y - minY) / (maxY - minY) * (img->height - 1)), (int)((x - minX) / (maxX - minX) * (img->width - 1)), colors[ labels[ i] % colors.size()]);
		}

		cvShowImage( "clusters", img);
		cvReleaseImage( &img);

		cout << second << endl;
		
		int ch = cvWaitKey(50);
		if( ch == 'j'){
			second--;
			if( second == -1){
				second = projectedData->cols - 1;
			}
		} else if( ch == 'k'){
			second = (second + 1) % projectedData->cols;
		} else if( ch == 'q'){
			break;
		}
		break;
	}
#endif
}

void TProjKmeans::generateRandomData( const vector< float *>& data, const int dimension)
{

	CvRNG rnd = cvRNG( -1);
	vector< CvMat *> means( 4);
	for( int i = 0; i < (int) means.size(); i++){
		means[ i] = cvCreateMat( 1, dimension, CV_32F);
		//cvZero( means[ i]);
		//cvSet1D( means[ i], 0, cvScalar( i * 4));
		cvRandArr( &rnd, means[ i], CV_RAND_NORMAL, cvScalar( 0), cvScalar( 1.0));
	}


	for( int i = 0; i < (int)data.size(); i++){
		CvMat randVec = cvMat( 1, dimension, CV_32F, (void *) data[ i]);
		cvRandArr( &rnd, &randVec, CV_RAND_NORMAL, cvScalar( 0), cvScalar( 1.0));
		cvAdd( &randVec, means[ i % means.size()], &randVec);
	}

	for( int i = 0; i < (int) means.size(); i++){
		cvReleaseMat( & means[ i]);
	}
}


void TProjKmeans::initLDP( const vector< float *> & data, const int dimension, CvMat *projection, const int words, 
			 const int samplesToEstimateCovariance,
			 const bool normDifferences, const bool myCovariance)
{
	assert( projection != NULL);
	assert( projection->cols == dimension);
	assert( projection->rows <= dimension);

	vector< int> labels( data.size());
		
	// init cluster centers randomly
	vector< int> initSamplesLeft( data.size());
	for( int i = 0; i < (int)data.size(); i++){
		initSamplesLeft[ i] = i;
	}

	CvMat *centersLong = cvCreateMat( words, dimension, CV_32F);
	CvMat *longRow = cvCreateMatHeader( 1, dimension, CV_32F);
	for( int i = 0; i < words; i++){

		const int id = rand() % initSamplesLeft.size();
		const int sampleID = initSamplesLeft[ id];

		initSamplesLeft[ id] = initSamplesLeft.back();
		initSamplesLeft.pop_back();

		CvMat sample = cvMat( 1, dimension, CV_32F, data[ i]);
		cvGetRow( centersLong, longRow, i);

		cvCopy( &sample, longRow);
	}
	cvReleaseMat( &longRow);

	CvFeatureTree *kdTree = cvCreateKDTree( centersLong);

	int nearestID;
	CvMat nearestMat = cvMat( 1, 1, CV_32S, &nearestID);
	double distance;
	CvMat distanceMat = cvMat( 1, 1, CV_64F, &distance);

	for( int i = 0; i < (int)data.size(); i++){

		CvMat sample = cvMat( 1, dimension, CV_32F, (void *)data[ i]);

		if( exactKNN){
			nearestID = findExactNN( &sample, centersLong);
		} else {
			cvFindFeatures( kdTree, &sample, &nearestMat, &distanceMat, 1, 200);
		}

		labels[ i] = nearestID;
	}
	cvReleaseFeatureTree( kdTree);
	cvReleaseMat( &centersLong);

	LDP( data, dimension, labels, projection, NULL, samplesToEstimateCovariance, normDifferences, myCovariance);
}

void TProjKmeans::projectData( const vector< float *> & data, CvMat *projection, CvMat *projectedData)
{

	CvMat * shortRow = cvCreateMatHeader( 1, projection->rows, CV_32F);

	CvMat * x = cvCreateMat( 1, projection->rows, CV_64F);
	CvMat * x2 = cvCreateMat( 1, projection->rows, CV_64F);
	CvMat * temp = cvCreateMat( 1, projection->rows, CV_64F);
	CvMat * L1 = cvCreateMat( 1, projection->rows, CV_64F);
	cvZero( x);
	cvZero( x2);
	cvZero( L1);


	for( int i = 0; i < (int)data.size(); i++){
		
		CvMat sample = cvMat( 1, projection->cols, CV_32F, data[ i]);
		
		cvGetRow( projectedData, shortRow, i);
		cvGEMM( &sample, projection, 1.0, NULL, 0.0, shortRow, CV_GEMM_B_T);

		cvConvert( shortRow, temp);
		cvAdd( temp, x, x);
		cvPow( temp, temp, 2);
		cvAdd( temp, x2, x2);
		cvConvert( shortRow, temp);
		cvAbs( temp, temp);
		cvAdd( temp, L1, L1);
	}

	cout << "Variance" << endl;
	for( int i = 0; i < x->cols; i++){
		cout << ' ' << sqrt( cvGetReal1D( x2, i) / data.size() - pow( cvGetReal1D( x, i) / data.size(), 2));
	}
	cout << endl;
	cout << "L1" << endl;
	for( int i = 0; i < x->cols; i++){
		cout << ' ' << cvGetReal1D( L1, i) / data.size();
	}
	cout << endl;
	cout << "Avg" << endl;
	for( int i = 0; i < x->cols; i++){
		cout << ' ' << cvGetReal1D( x, i) / data.size();
	}
	cout << endl;
	cvReleaseMat( &x);
	cvReleaseMat( &x2);
	cvReleaseMat( &temp);
	cvReleaseMat( &L1);

	cvReleaseMat( &shortRow);
}

TProjKmeans::TProjKmeans( const std::vector< float *>& data, const int dimension, const int words, const int maxIteration, const bool _exactKNN)
	: statusOK( true), projection( NULL), centers( cvCreateMat( words, dimension, CV_32F)), tree( NULL), projectedPoint( NULL), exactKNN( _exactKNN), inputVectorDimension( dimension), avgWordDistance( 0.0)
{

	// init cluster centers randomly
	vector< int> initSamplesLeft( data.size());
	for( int i = 0; i < (int)data.size(); i++){
		initSamplesLeft[ i] = i;
	}

	CvMat * row1 = cvCreateMatHeader( 1, dimension, CV_32F);
	for( int i = 0; i < words; i++){

		const int id = rand() % initSamplesLeft.size();
		const int sampleID = initSamplesLeft[ id];

		initSamplesLeft[ id] = initSamplesLeft.back();
		initSamplesLeft.pop_back();

		cvGetRow( centers, row1, i);

		CvMat row2 = cvMat( 1, dimension, CV_32F, data[ sampleID]);

		cvCopy( &row2, row1);
	}

	vector< int> clusterLabels( data.size(), -1);

	for( int iteration = 0; iteration < maxIteration; iteration++){

		// compute new labels
		tree = cvCreateKDTree( centers);

		int changeCount = 0;
		int nearestID;
		CvMat nearestMat = cvMat( 1, 1, CV_32S, &nearestID);
		double distance;
		CvMat distanceMat = cvMat( 1, 1, CV_64F, &distance);

		map< int, vector<int> > labelMap;
		{
			clock_t time_start = clock();
			for( int i = 0; i < (int)data.size(); i++){

				CvMat sample = cvMat( 1, dimension, CV_32F, data[ i] );

				if( exactKNN){
					nearestID = findExactNN( &sample, centers);
				} else {
					cvFindFeatures( tree, &sample, &nearestMat, &distanceMat, 1, 200);
				}

				changeCount += (int) (clusterLabels[ i] != nearestID);
				clusterLabels[ i] = nearestID;
				labelMap[ nearestID].push_back( i);
			}
			cout << "DONE KNN in (sec): " <<  (double)( clock() - time_start) / CLOCKS_PER_SEC << endl;
		}

		cvReleaseFeatureTree( tree); tree = NULL;
		cout << "Cluster change ratio: " << changeCount / (double) data.size() << endl;


		// compute new cluster centers
		cvSet( centers, cvScalar( 0));
		for( map< int, vector< int> >::const_iterator labelIt = labelMap.begin(); labelIt != labelMap.end(); labelIt++){

			cvGetRow( centers, row1, labelIt->first);

			for( int i  = 0; i < (int)labelIt->second.size(); i++){
				CvMat sample = cvMat( 1, dimension, CV_32F, data[ labelIt->second[ i]] );
				cvAdd( row1, &sample, row1); 
			}

			cvScale( row1, row1, 1.0 / labelIt->second.size());
		}

		//reinit lost clusters
		int lostClusterCount = 0;
		for( int i = 0; i < centers->rows; i++){
			if( labelMap.find( i) == labelMap.end()){

				lostClusterCount++;

				const int sampleID = rand() % data.size();

				cvGetRow( centers, row1, i);

				CvMat row2 = cvMat( 1, dimension, CV_32F, data[ sampleID]);

				cvCopy( &row2, row1);

			}
		}
		cout << "Lost clusters: " << lostClusterCount << endl;

	}

	cvReleaseMat( &row1);
	tree = cvCreateKDTree( centers);
}



TProjKmeans::TProjKmeans( const std::vector< float *>& data, const int dimension, 
						 const std::vector< int> labels,  const int targetDim, 
						 const int words, const int maxIteration, const bool doPCA,
						 const bool _exactKNN)
	: statusOK( true), projection( cvCreateMat( targetDim, dimension, CV_32F)), centers( cvCreateMat( words, targetDim, CV_32F)), 
	tree( NULL), projectedPoint( cvCreateMat( 1, targetDim, CV_32F)),
	exactKNN( _exactKNN), inputVectorDimension( dimension), avgWordDistance( 0.0)

{

//	generateRandomData( data, dimension);

#if SHOW_DATA
	cvNamedWindow( "clusters");
#endif

	if( doPCA){
		PCA( data, dimension, projection, NULL, samplesToEstimateCovariance);
	} else {
		LDP( data, dimension, labels, projection, NULL, samplesToEstimateCovariance, normDifferences, myCovariance);
	}

	CvMat *projectedData = cvCreateMat( data.size(), targetDim, CV_32F);

	projectData( data, projection, projectedData);

	// init cluster centers randomly
	vector< int> initSamplesLeft( data.size());
	for( int i = 0; i < (int)data.size(); i++){
		initSamplesLeft[ i] = i;
	}

	CvMat * shortRow1 = cvCreateMatHeader( 1, targetDim, CV_32F);
	CvMat * shortRow2 = cvCreateMatHeader( 1, targetDim, CV_32F);
	for( int i = 0; i < words; i++){

		const int id = rand() % initSamplesLeft.size();
		const int sampleID = initSamplesLeft[ id];

		initSamplesLeft[ id] = initSamplesLeft.back();
		initSamplesLeft.pop_back();

		cvGetRow( centers, shortRow1, i);
		cvGetRow( projectedData, shortRow2, sampleID);

		cvCopy( shortRow2, shortRow1);
	}

	vector< int> clusterLabels( data.size(), -1);

	for( int iteration = 0; iteration < maxIteration; iteration++){

		// compute new labels
		tree = cvCreateKDTree( centers);

		int changeCount = 0;
		int nearestID;
		CvMat nearestMat = cvMat( 1, 1, CV_32S, &nearestID);
		double distance;
		CvMat distanceMat = cvMat( 1, 1, CV_64F, &distance);

		map< int, vector<int> > labelMap;
		{
			clock_t time_start = clock();
			for( int i = 0; i < (int)data.size(); i++){

				CvMat sample = cvMat( 1, targetDim, CV_32F, cvPtr2D( projectedData, i, 0) );

				if( exactKNN){
					nearestID = findExactNN( &sample, centers);
				} else {
					cvFindFeatures( tree, &sample, &nearestMat, &distanceMat, 1, 200);
				}

				changeCount += (int) (clusterLabels[ i] != nearestID);
				clusterLabels[ i] = nearestID;
				labelMap[ nearestID].push_back( i);
			}
			cout << "DONE KNN in (sec): " <<  (double)( clock() - time_start) / CLOCKS_PER_SEC << endl;
		}

		cvReleaseFeatureTree( tree); tree = NULL;
		cout << "Cluster change ratio: " << changeCount / (double) data.size() << endl;

		showData( projectedData, clusterLabels);

		// compute new cluster centers
		cvSet( centers, cvScalar( 0));
		for( map< int, vector< int> >::const_iterator labelIt = labelMap.begin(); labelIt != labelMap.end(); labelIt++){

			cvGetRow( centers, shortRow1, labelIt->first);

			for( int i  = 0; i < (int) labelIt->second.size(); i++){
				cvGetRow( projectedData, shortRow2, labelIt->second[ i]);
				cvAdd( shortRow1, shortRow2, shortRow1); 
			}

			cvScale( shortRow1, shortRow1, 1.0 / labelIt->second.size());
		}

		//reinit lost clusters
		int lostClusterCount = 0;
		for( int i = 0; i < centers->rows; i++){
			if( labelMap.find( i) == labelMap.end()){

				lostClusterCount++;

				const int sampleID = rand() % data.size();

				cvGetRow( centers, shortRow1, i);
				cvGetRow( projectedData, shortRow2, sampleID);

				cvCopy( shortRow2, shortRow1);
			}
		}
		cout << "Lost clusters: " << lostClusterCount << endl;

	}

	cvReleaseMat( &shortRow1);
	cvReleaseMat( &shortRow2);
	cvReleaseMat( &projectedData);

	tree = cvCreateKDTree( centers);
}

TProjKmeans::TProjKmeans( const vector< float *>& data, const int dimension, const int targetDim, const int words, const int maxIteration, const bool _exactKNN)
	: statusOK( true), projection( cvCreateMat( targetDim, dimension, CV_32F)), centers( cvCreateMat( words, targetDim, CV_32F)), 
	tree( NULL), projectedPoint( cvCreateMat( 1, targetDim, CV_32F)),
	exactKNN( _exactKNN), inputVectorDimension( dimension), avgWordDistance( 0.0)
{

//	generateRandomData( data, dimension);

#if SHOW_DATA
	cvNamedWindow( "clusters");
#endif

	// init projection on random clusters in the original space (we dont have any projection yet - cant go to reduced space)
	vector< int> labels( data.size());

	CvMat *projectedData = cvCreateMat( data.size(), targetDim, CV_32F);

	// init projection with LDP
	initLDP( data, dimension, projection, words, samplesToEstimateCovariance, normDifferences, myCovariance);

	projectData( data, projection, projectedData);

	// init cluster centers randomly
	vector< int> initSamplesLeft( data.size());
	for( int i = 0; i < (int)data.size(); i++){
		initSamplesLeft[ i] = i;
	}

	CvMat * shortRow1 = cvCreateMatHeader( 1, targetDim, CV_32F);
	CvMat * shortRow2 = cvCreateMatHeader( 1, targetDim, CV_32F);
	for( int i = 0; i < words; i++){

		const int id = rand() % initSamplesLeft.size();
		const int sampleID = initSamplesLeft[ id];

		initSamplesLeft[ id] = initSamplesLeft.back();
		initSamplesLeft.pop_back();

		cvGetRow( centers, shortRow1, i);
		cvGetRow( projectedData, shortRow2, sampleID);

		cvCopy( shortRow2, shortRow1);
	}


	for( int iteration = 0; iteration < maxIteration; iteration++){

		// compute new labels
		CvFeatureTree *kdTree = cvCreateKDTree( centers);

		int changeCount = 0;
		int nearestID;
		CvMat nearestMat = cvMat( 1, 1, CV_32S, &nearestID);
		double distance;
		CvMat distanceMat = cvMat( 1, 1, CV_64F, &distance);

		map< int, vector<int> > labelMap;
		clock_t time_start = clock();
		for( int i = 0; i < (int)data.size(); i++){

			CvMat sample = cvMat( 1, targetDim, CV_32F, cvPtr2D( projectedData, i, 0) );

			if( exactKNN){
				nearestID = findExactNN( &sample, centers);
			} else {
				cvFindFeatures( kdTree, &sample, &nearestMat, &distanceMat, 1, 200);
			}

			changeCount += (int) (labels[ i] != nearestID);
			labels[ i] = nearestID;
			labelMap[ nearestID].push_back( i);
		}
		cout << "KNN end in (sec): " <<  (double)( clock() - time_start) / CLOCKS_PER_SEC << endl;
		cvReleaseFeatureTree( kdTree);
		cout << "Cluster change ratio: " << changeCount / (double) data.size() << endl;

		/*for( map< int, vector< int> >::const_iterator labelIt = labelMap.begin(); labelIt != labelMap.end(); labelIt++){
			cout << labelIt->first << ':' << labelIt->second.size() << ' ';
		}
		cout << endl;*/

		showData( projectedData, labels);

		// compute LDP
		LDP( data, dimension, labels, projection, NULL, samplesToEstimateCovariance, normDifferences, myCovariance);
	
		projectData( data, projection, projectedData);

		showData( projectedData, labels);

		// compute new cluster centers
		cvSet( centers, cvScalar( 0));
		for( map< int, vector< int> >::const_iterator labelIt = labelMap.begin(); labelIt != labelMap.end(); labelIt++){

			cvGetRow( centers, shortRow1, labelIt->first);

			for( int i  = 0; i < (int)labelIt->second.size(); i++){
				cvGetRow( projectedData, shortRow2, labelIt->second[ i]);
				cvAdd( shortRow1, shortRow2, shortRow1); 
			}

			cvScale( shortRow1, shortRow1, 1.0 / labelIt->second.size());
		}

		//reinit lost clusters
		int lostClusterCount = 0;
		for( int i = 0; i < centers->rows; i++){
			if( labelMap.find( i) == labelMap.end()){

				lostClusterCount++;

				const int sampleID = rand() % data.size();

				cvGetRow( centers, shortRow1, i);
				cvGetRow( projectedData, shortRow2, sampleID);

				cvCopy( shortRow2, shortRow1);
			}
		}
		cout << "Lost clusters: " << lostClusterCount << endl;
	}

	cvReleaseMat( &shortRow1);
	cvReleaseMat( &shortRow2);

	tree = cvCreateKDTree( centers);
}


/*

void LDA{

	if( data.empty()){
		return;
	}

	ldaDim = vector< vector< float> >( data[ 0]->cols, vector< float>( data[ 0]->cols, 0.0f));

	// get the number of positive an negative samples
	int posSize = 0;
	int negSize = 0;
	for( unsigned i = 0; i < data.size(); i++){
		if( classID[ i] > 0)
        {
			posSize++;
		} else {
			negSize++;
		}
	}
	
	
	// divide positive and negative samples into separate sets
	CvMat **posSet = new CvMat *[ posSize];
	CvMat **negSet = new CvMat *[ negSize];

	int posPosition = 0;
	int negPosition = 0;

	for( unsigned currentSampleID = 0; currentSampleID != data.size(); currentSampleID++){
        if( classID[ currentSampleID] > 0){
			posSet[ posPosition++] = data[ currentSampleID];
		} else {
			negSet[ negPosition++] = data[ currentSampleID];
		}
	}

	// compute mean vectors and covariance matrices for positive and negative sets
	CvMat *posCov = cvCreateMat( (int) ldaDim.size(), (int) ldaDim.size(), CV_64FC1);
	CvMat *negCov = cvCreateMat( (int) ldaDim.size(), (int) ldaDim.size(), CV_64FC1);

	CvMat *posMean = cvCreateMat( 1, (int) ldaDim.size(), CV_64FC1);
	CvMat *negMean = cvCreateMat( 1, (int) ldaDim.size(), CV_64FC1);

	cvCalcCovarMatrix( (const CvArr **)posSet, posSize, posCov, posMean, CV_COVAR_NORMAL);
	cvCalcCovarMatrix( (const CvArr **)negSet, negSize, negCov, negMean, CV_COVAR_NORMAL);

	// the separate positive and negative sets will not be needed anymore (all information is contained in the mean vectors and covariance matrices)
	delete[] posSet;
	delete[] negSet;


	CvMat *show = cvCreateMat( (int) ldaDim.size(), (int) ldaDim.size(), CV_64FC1);
	CvMat *tempMM = cvCreateMat( (int) ldaDim.size(), (int) ldaDim.size(), CV_64FC1);

#if showLDAComputations
	cout << "BEGIN" << endl;
	cvWaitKey( 10000);
#endif

#if showLDAComputations
	cvNormalize( posCov, show, 0.0, 1.0, CV_MINMAX);
	cout << "Positive Covariant Matrix" << endl;
	cvShowImage( "XXX", show);
	cvWaitKey( 10000);
#endif

#if showLDAComputations
	cvNormalize( negCov, show, 0.0, 1.0, CV_MINMAX);
	cout << "Negative Covariant Matrix" << endl;
	cvShowImage( "XXX", show);
	cvWaitKey( 10000);
#endif


	CvMat *Sw = cvCreateMat( (int) ldaDim.size(), (int) ldaDim.size(), CV_64FC1);
	cvAdd( posCov, negCov, Sw);
	cvConvertScale( Sw, Sw, 1.0 / 2.0);

#if showLDAComputations
	cvNormalize( Sw, show, 0.0, 1.0, CV_MINMAX);
	cout << "Sw" << endl;
	cvShowImage( "XXX", show);
	cvWaitKey( 10000);
#endif

	CvMat *means[] = { posMean, negMean};
	CvMat *Sb = cvCreateMat( (int) ldaDim.size(), (int) ldaDim.size(), CV_64FC1);
	CvMat *temp = cvCreateMat( 1, (int) ldaDim.size(), CV_64FC1);
	cvCalcCovarMatrix( (const CvArr **)means, 2, Sb, temp, CV_COVAR_NORMAL);

#if showLDAComputations
	cvNormalize( Sb, show, 0.0, 1.0, CV_MINMAX);
	cout << "Sb" << endl;
	cvShowImage( "XXX", show);
	cvWaitKey( 10000);
#endif 

	CvMat *criterion = cvCreateMat( (int) ldaDim.size(), (int) ldaDim.size(), CV_64FC1);

	cvInvert( Sw, criterion);
#if showLDAComputations
	cvNormalize( criterion, show, 0.0, 1.0, CV_MINMAX);
	cout << "criterion (invert Sw)" << endl;
	cvShowImage( "XXX", show);
	cvWaitKey( 10000);
#endif

	cvGEMM( criterion, Sb, 1.0, NULL, 0.0, tempMM);
	cvConvertScale( tempMM, Sb);
#if showLDAComputations
	cvNormalize( Sb, show, 0.0, 1.0, CV_MINMAX);
	cout << "after cvGEEM" << endl;
	cvShowImage( "XXX", show);
	cvWaitKey( 10000);
#endif

	CvMat *eigenVectors = cvCreateMat( (int) ldaDim.size(), (int) ldaDim.size(), CV_64FC1);

	// compute the eigenvectors using SVD
	cvSVD( criterion,  temp, eigenVectors, NULL, CV_SVD_MODIFY_A | CV_SVD_U_T );
#if showLDAComputations	
	cvNormalize( eigenVectors, show, 0.0, 1.0, CV_MINMAX);
	cout << "eigenVectors" << endl;
	cvShowImage( "XXX", show);
	cvWaitKey( 10000);
#endif

	// sort the eigenvectors 
	vector< int> sortedEV( ldaDim.size(), 0) ;

	cout << endl;
	for( unsigned i = 0; i < ldaDim.size(); i++){
        
		double bestValue = 1000000000000000000.0;
		int bestPosition = 0;
		for( unsigned j = 0; j < ldaDim.size(); j++){
			if( bestValue > cvGetReal1D( temp, j)){
				bestValue = cvGetReal1D( temp, j);
				bestPosition = j;
			}
		}

		cout << bestPosition << ' ' << bestValue << endl;

		cvSetReal1D( temp, bestPosition, 1000000000000000000.0);
		sortedEV[ i] = bestPosition;
		
	}

	// copy the eigenvectors
	for( unsigned row = 0; row < ldaDim.size(); row++){
		for( unsigned column = 0; column < ldaDim.size(); column++){
			ldaDim[ row][ column] = (float) cvGetReal2D( eigenVectors, sortedEV[ row], column);
		}
	}

	cvReleaseMat( &posCov);
	cvReleaseMat( &negCov);
	cvReleaseMat( &posMean);
	cvReleaseMat( &negMean);
	cvReleaseMat( &show);
	cvReleaseMat( &tempMM);
	cvReleaseMat( &Sw);
	cvReleaseMat( &Sb);
	cvReleaseMat( &temp);
	cvReleaseMat( &criterion);
	cvReleaseMat( &eigenVectors);

}*/


