#include "TVFeatureVectors.h"

#include <cfloat>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>

namespace TRECVID{

using namespace std;

map< string, TNormalization> normalizationTranslation;
map< TNormalization, string> invNormalizationTranslation;

map< string, TKernel> kernelTranslation;
map< TKernel, string> invKernelTranslation;

void prepareTranslations(){
	kernelTranslation[ "X2"] = K_X2;
	kernelTranslation[ "LINEAR"] = K_LINEAR;
	kernelTranslation[ "L1"] = K_L1;
	kernelTranslation[ "L2"] = K_L2;
	for( std::map< string, TKernel>::const_iterator it = kernelTranslation.begin(); it != kernelTranslation.end(); it++){
		invKernelTranslation[ it->second] = it->first;
	}

	normalizationTranslation[ ""] = NO_NORMALIZATION;
	normalizationTranslation[ "NONE"] = NO_NORMALIZATION;
	normalizationTranslation[ "L1"] = L1_NORMALIZATION;
	normalizationTranslation[ "L2"] = L2_NORMALIZATION;
	for( map< string, TNormalization>::const_iterator it = normalizationTranslation.begin(); it != normalizationTranslation.end(); it++){
		invNormalizationTranslation[ it->second] = it->first;
	}
}

void normalize( TFeatureVector * featureVector, const TNormalization normalizationType){
	if( featureVector == NULL || normalizationType == NO_NORMALIZATION){
		return ;
	} else if( normalizationType == L1_NORMALIZATION){
		featureVector->normalizeL1();
	} else if( normalizationType == L2_NORMALIZATION){
		featureVector->normalizeL2();
	} else {
		cerr << "Error: Unknown normalization type." << endl;
	}
}


const std::string TFeatureVectorFile::fileHead = "FeatureVectorFile2.0";

#define FV_DATA_8b 2

void TFeatureVectorFile::writeVector( std::ofstream *outStream, const TFeatureVector *fv)
{
	assert( outStream != NULL);

	// is it here?
	char some = fv != NULL;

	if( !some){
		outStream->write( &some, sizeof( char));
	} else {
		
		// check if it fits to unsigned 8b
		bool goodFor8b = true;
		for( int i = 0; i < ((fv->dense) ? ( fv->dimension): ( fv->IDLength)); i++){
			if( fv->data[ i] < 0 || fv->data[ i] > 255 || fv->data[ i] - ((int) fv->data[ i]) != 0){
				goodFor8b = false;
				break;
			}
		}
		
		if( goodFor8b){
			some = FV_DATA_8b;
		}

		outStream->write( &some, sizeof( char));
		fv->write( outStream, goodFor8b);
	}
}

void TFeatureVectorFile::readVector( std::ifstream *inStream, TFeatureVector *&fv)
{
	assert( inStream != NULL);

	// is it here?
	char some;
	inStream->read( &some, sizeof( char));

	if( some){
		fv = new TFeatureVector( inStream, some == FV_DATA_8b);
	} else {
		fv = NULL;
	}
}


void TFeatureVectorFile::writeToFile( const std::string filename, const std::vector< TFeatureVector *> &vectors)
{
	TFeatureVectorFile file( filename, true, vectors.size());
	file.add( vectors);
}

void  TFeatureVectorFile::readFromFile( const std::string filename, std::vector< TFeatureVector *> &vectors)
{
	TFeatureVectorFile file( filename, false);
	file.readAll( vectors);
}


TFeatureVectorFile::~TFeatureVectorFile()
{
	delete inStream;

	if( outStream != NULL){

		outStream->seekp( 0);

		outStream->write( fileHead.data(), fileHead.size() * sizeof( char));

		// write number of vectors
		long long temp = indexPos;
		outStream->write( (char *) &temp, sizeof( long long));

		// write the index
		outStream->write( (char *) & index[ 0], sizeof( long long) * (std::streamsize) indexPos);

		delete outStream;
	}
}

TFeatureVectorFile::TFeatureVectorFile( const std::string _filename, const bool forWriting, long long maxSize)
	: filename( _filename), inStream( NULL), outStream( NULL), indexPos( 0)
{
    if( forWriting){

        index = vector< long long>( (unsigned int) ((maxSize > 0)? (maxSize) : (defMaxSize)), -1);

        outStream = new ofstream( filename.data(), ios::binary | ios::trunc);

	    if( !outStream->good()){
		    cerr << "ERROR: Unable to open file \"" << filename << "\" for writing feature vectors." << endl;
		    throw string( "ERROR: Unable to open file for writing feature vectors");
	    }

		outStream->seekp( fileHead.size() * sizeof( char));

	    // write number of vectors
	    long long temp = 0;
	    outStream->write( (char *) &temp, sizeof( long long));

	    // seek to the end of the index
	    outStream->seekp( index.size() * sizeof( long long), ofstream::cur);

    } else {

        inStream = new ifstream( _filename.data(), ios::binary);

	    if( !inStream->good()){
		    cerr << "ERROR: Unable to open file \"" << filename << "\" for reading feature vectors." << endl;
		    throw string( "ERROR: Unable to open file for reading feature vectors");
	    }

	    // check the file type
	    char buffer[ 500];
	    buffer[ fileHead.size()] = 0;
	    inStream->read( buffer, fileHead.size() * sizeof( char));

	    if( fileHead != buffer){
		    cerr << "ERROR: This is not a feature vector file \"" << filename << "\"." << endl;
		    throw string( "ERROR: Not a feature vector file");
	    }

	    // read number of vectors
	    long long temp = 0;
	    inStream->read( (char *) &temp, sizeof( long long));

	    // read index
	    index.resize( (unsigned int)temp);
		if( !index.empty()){
			long long *ptr = &index[ 0];
			inStream->read( (char *)ptr, sizeof( long long) * index.size());
		}
    }
}

void TFeatureVectorFile::add( const std::vector< TFeatureVector *> &vectors)
{
	// write all vectors
	for( unsigned int i = 0; i <  vectors.size(); i++){
		add( vectors[ i]);
	}
}

void TFeatureVectorFile::add( const TFeatureVector * fv)
{
	if( indexPos >= (long long)index.size()){
		cerr << "ERROR: Exceding allocated index size (" << index.size() << ") when writing feature vectors to file \"" << filename << "\"." << endl;
		throw string( "ERROR: Exceding allocated index size  when writing feature vectors to file .");
	}

#ifdef SPECIFIC_LF
	index[ (unsigned int) indexPos] = outStream->tellp().seekpos();	
	if( index[ (unsigned int)indexPos] < 0){
		cerr << "XXXXXXXXXXXXXXXXXXXXXXX." << endl;
		exit( -1);
	}
	indexPos++;
#else
	index[ (unsigned int)indexPos] = outStream->tellp();		
	if( index[ (unsigned int)indexPos] < 0){
		cerr << "ERROR: Unable to write files larger than 2GB. Work on different machine or define SPECIFIC_LF (does not work with all compilers)." << endl;
		exit( -1);
	}
	indexPos++;
#endif


	writeVector( outStream, fv);
}

void TFeatureVectorFile::readAll( std::vector< TFeatureVector *> &vectors)
{
	vectors.resize( index.size());

	for( int i = 0; i < (int) index.size(); i++){
		read( vectors[ i], i);
	}
}

long long TFeatureVectorFile::getVectorCount( )
{
    return index.size();
}

void TFeatureVectorFile::read( std::vector< TFeatureVector *> &vectors, const std::vector< long long>& IDsToRead)
{
	vectors.resize( index.size());

	for( int i = 0; i < (int) IDsToRead.size(); i++){
		read( vectors[ (unsigned int)IDsToRead[ i]], IDsToRead[ i]);
	}
}

void TFeatureVectorFile::read( TFeatureVector * &fv, const long long IDToRead, const bool checkPresence)
{
	assert( IDToRead < index.size());

	if( numeric_limits< streamoff>::max() < index[ (unsigned int)IDToRead]){
		cerr << "Error: Unable to read files larger then 2GB. Run this on 64bit machine." << endl;
		throw string( "Error: Unable to read files larger then 2GB. Run this on 64bit machine."); 
	}

	if( inStream->seekg( index[ (unsigned int)IDToRead]).fail()){
		cerr << "Error: failed to seek to position " << index[ (unsigned int) IDToRead] << " when reading feature vector " << IDToRead << "." << endl;
		throw string( "Error: failed to seek.");
	};

	if( index[ (unsigned int)IDToRead] != inStream->tellg()){
		cerr << "ERROR: " << index[ (unsigned int)IDToRead] << ' ' <<  inStream->tellg() << endl;
		throw string( "Error: failed to seek");
	}

	inStream->seekg( index[ (unsigned int)IDToRead]);

	readVector( inStream, fv);

	if( checkPresence && fv == NULL){
		cerr << "Error: Feature vector " << IDToRead << " is not present in file " << this->filename << "." << endl;
		throw string( "Error: missing feature vector in feature vector file.");
	}
}











void TFeatureVector::toDense()
{
	if( dense){
		return;
	}

	float *newData = new float[ dimension];

	for( int i = 0; i < dimension; i++){
		newData[ i] = 0;
	}

	for( int i = 0; i < IDLength; i++){
		assert( IDs[ i] < dimension);
		newData[ IDs[ i]] = data[ i];
	}

	delete[] data;
	delete[] IDs;

	data = newData;
	IDs = NULL;
	IDLength = 0;
	dense = true;
}

void TFeatureVector::toSparse()
{
	if( !dense){
		return;
	}


	IDLength = 0;

	for( int i = 0; i < dimension; i++){
		if( data[ i] != 0.0f){
			IDLength++;
		}
	}

	IDs = new int32_t[ IDLength + 1];
	IDs[ IDLength] = INT_MAX;
	float *newData = new float[ IDLength];

	int ID = 0;

	for( int i = 0; i < dimension; i++){
		if( data[ i] != 0.0f){
			newData[ ID] = data[ i];
			IDs[ ID] = i;
			ID++;
		}
	}

	delete[] data;
	data = newData;
    dense = false;
}

void TFeatureVector::optimize()
{
	// check if it fits to unsigned 8b
	bool goodFor8b = true;
	for( int i = 0; i < ( dense ? dimension : IDLength); i++){
		if( data[ i] < 0 || data[ i] > 255 || data[ i] - ((int)data[ i]) != 0){
			goodFor8b = false;
			break;
		}
	}

	int dataSize = goodFor8b ? sizeof( unsigned char) : sizeof( float);

	const int denseSize = dataSize * dimension;
	int sparseSize = 0;

	if( dense){
		for( int i = 0; i < dimension; i++){
			if( data[ i] != 0.0f){
				sparseSize += sizeof( int32_t) + dataSize;
			}
		}
	} else {
		sparseSize = ( sizeof( int32_t) + dataSize) * IDLength + sizeof( int32_t);
	}

	if( denseSize > sparseSize){
		this->toSparse();
	} else {
		this->toDense();
	}
}


void TFeatureVector::write( std::ofstream *outStream, const bool data8b) const
{
	char d = dense;

	outStream->write( &d, sizeof( char));
	outStream->write( (char *) &dimension, sizeof( int32_t));

	if( !dense){
		outStream->write( (char *) &IDLength, sizeof( int32_t));
	}


	const int dataLength( ( dense)? ( dimension) : ( IDLength));

	if( data8b){
		unsigned char *tempData = new unsigned char[ dataLength];
		
		for( int i = 0; i < dataLength; i++){
			tempData[ i] = (unsigned char)data[ i];
		}
		outStream->write( (char *) tempData, sizeof( unsigned char) * dataLength);

		delete[] tempData;

	} else {
		outStream->write( (char *) data, sizeof( float) * dataLength);
	}

	if( !dense){
		outStream->write( (char *) IDs, sizeof( int32_t) * IDLength);
	}
}

TFeatureVector::TFeatureVector( std::ifstream *inStream, const bool data8b) 
	: dense( true), data( NULL), IDs( NULL), dimension( 0), IDLength( 0)
{

	char d;
	inStream->read( &d, sizeof( char));
	dense = d;

	inStream->read( (char *) &dimension, sizeof( int32_t));

	if( !dense){
		inStream->read( (char *) &IDLength, sizeof( int32_t));
	}


	const int dataLength = ( dense)? ( dimension) : ( IDLength);

	data = new float[ dimension];

	if( data8b){
		unsigned char *tempData = new unsigned char[ dataLength];
		
		inStream->read( (char *) tempData, sizeof( unsigned char) * dataLength);
		for( int i = 0; i < dataLength; i++){
			data[ i] = tempData[ i];
		}

		delete[] tempData;	
	} else {
		inStream->read( (char *) data, sizeof( float) * dataLength);
	}

	if( !dense){
		IDs = new int32_t[ IDLength + 1];
		IDs[ IDLength] = INT_MAX;
		inStream->read( (char *) IDs, sizeof( int32_t) * IDLength);
	}
}

TFeatureVector::TFeatureVector( const int32_t _dimension)
	: dense( true), data( new float[ _dimension]), IDs( NULL), dimension( _dimension), IDLength( 0)
{
    for( int i = 0; i < dimension; i++){
        data[ i] = 0.0f;
    }
}

TFeatureVector::TFeatureVector( const TFeatureVector& src)
    :dense( src.dense), data( NULL), IDs( NULL), dimension( src.dimension), IDLength( src.IDLength)
{
    if( dense){
        data = new float[ dimension];
        memcpy( data, src.data, sizeof( float) * dimension);
    } else {
        data = new float[ IDLength];
        memcpy( data, src.data, sizeof( float) * IDLength);
		IDs = new int32_t[ IDLength + 1];
		IDs[ IDLength] = INT_MAX;
        memcpy( IDs, src.IDs, sizeof( int32_t) * IDLength);
    }
}


TFeatureVector & TFeatureVector::operator=( const TFeatureVector& src)
{
    if( this != &src){
        if( dense){
            
            float *tempData = new float[ src.dimension];
            memcpy( tempData, src.data, sizeof( float) * src.dimension);

            delete[] data;
            delete[] IDs;

            dense = true;
            data = tempData;
            IDs = NULL;
            dimension = src.dimension;
            IDLength = src.IDLength;

        } else {

            float *tempData = new float[ src.IDLength];
            memcpy( tempData, src.data, sizeof( float) * src.IDLength);
		    
            int32_t *tempIDs = new int32_t[ src.IDLength + 1];
		    tempIDs[ src.IDLength] = INT_MAX;
            memcpy( tempIDs, src.IDs, sizeof( int32_t) * src.IDLength);

            delete[] data;
            delete[] IDs;

            dense = false;
            data = tempData;
            IDs = tempIDs;
            dimension = src.dimension;
            IDLength = src.IDLength;
        }
    }
        
    return *this;
}

TFeatureVector & TFeatureVector::concatenate( const TFeatureVector & fv2)
{

    TFeatureVector fv( fv2);

    if( this->dense){

        fv.toDense();

        float *tempData = new float[ this->dimension + fv.dimension];
        memcpy( tempData, this->data, sizeof( float) * this->dimension);
        memcpy( tempData + this->dimension, fv.data, sizeof( float) * fv.dimension);

        delete[] data;
        delete[] IDs;

        data = tempData;
        IDs = NULL;
        dimension = this->dimension + fv.dimension;
        IDLength = 0;

    } else {

        fv.toSparse();

        float *tempData = new float[ this->IDLength + fv.IDLength];
        memcpy( tempData, this->data, sizeof( float) * this->IDLength);
        memcpy( tempData + this->IDLength, fv.data, sizeof( float) * fv.IDLength);
	    
        int32_t *tempIDs = new int32_t[ this->IDLength + fv.IDLength + 1];
	    tempIDs[ this->IDLength + fv.IDLength ] = INT_MAX;
        memcpy( tempIDs, this->IDs, sizeof( int32_t) * this->IDLength);
        memcpy( tempIDs + this->IDLength, fv.IDs, sizeof( int32_t) * fv.IDLength);

        delete[] data;
        delete[] IDs;

        data = tempData;
        IDs = tempIDs;
        dimension = this->dimension + fv.dimension;
        IDLength = this->IDLength + fv.IDLength;
    }

    return *this;
}

void TFeatureVector::normalizeL1( const float length)
{
    const int maxIndex = (dense)? (dimension) : ( IDLength);

    double sum = 0.0;
    for( int i = 0; i < maxIndex; i++){
        sum += fabsf( data[ i]);
    }

    const float normFactor = length / (float) sum;

    for( int i = 0; i < maxIndex; i++){
	    data[ i] *= normFactor;
    }
}

void TFeatureVector::normalizeL2( const float length)
{
    const int maxIndex = (dense)? (dimension) : ( IDLength);

    double sum = 0.0;
    for( int i = 0; i < maxIndex; i++){
	    sum += data[ i] * data[ i];
    }

    const float normFactor = length / sqrtf( (float) sum);

    for( int i = 0; i < maxIndex; i++){
	    data[ i] *= normFactor;
    }
}


void TFeatureVector::remDim( const vector< int> &removeDim)
{
	this->toDense();

	// remove some dimensions
	if( !removeDim.empty()){

		if( removeDim.back() >= this->dimension){
			cerr << "Error: Should remove dimension " << removeDim.back() << ", but lenght of the vector is only " << this->dimension << "." << endl;
			throw string( "Dimension to remove excedes length of the feature vector.");
		}

		const int newDim = this->dimension - removeDim.size();
		float *newData = new float[ newDim];
		int newPos = 0;
		int remPos = 0;
		for( int oldPos = 0; oldPos < this->dimension; oldPos++){
			if( remPos < (int) removeDim.size() && removeDim[ remPos] == oldPos){
				remPos++;
			} else{
				newData[ newPos++] = this->data[ oldPos];
			}
		}
		delete this->data;
		this->data = newData;
		this->dimension = newDim;
	}
}
	
void TFeatureVector::scaleDim( const vector< pair< int, float> > &scales)
{
	// scale specified dimensions
	for( int i = 0; i < (int)scales.size(); i++){
		if( scales[ i].first < this->dimension){
			this->data[ scales[ i].first] *= scales[ i].second;
		}
	}
}


void TFeatureVector::normalizeStdDev( const float stdDev)
{
    const int maxIndex = (dense)? (dimension) : ( IDLength);

    double sum = 0.0;
    double sum2 = 0.0;

    for( int i = 0; i < maxIndex; i++){
	    sum += data[ i];
	    sum2 += data[ i] * data[ i];
    }

    sum /= maxIndex;

    const double normFactor = stdDev / sqrt( sum2 / maxIndex - pow( sum, 2));

    for( int i = 0; i < maxIndex; i++){
	    data[ i] = (float)( (data[ i] - sum) * normFactor);
    }
}


float TFeatureVector::distX2( const TFeatureVector &fv2) const
{
    float sum = 0;

    if( dense){

        assert( fv2.dense);

        for( int i = 0; i < dimension; i++){
		    
            const float val1 = this->data[ i];
		    const float val2 = fv2.data[ i];
			if( val1 + val2 != 0){
				sum += (val1 - val2) * (val1 - val2) / ( val1 + val2);
			}
        }

    } else {

        assert( ! fv2.dense);

	    int index1 = 0;
	    int index2 = 0;
        int size1 = this->IDLength - 1;
        int size2 = fv2.IDLength - 1;

	    while( index1 < size1 || index2 < size2){

		    if( this->IDs[ index1] == fv2.IDs[ index2]){

			    const float val1 = this->data[ index1];
			    const float val2 = fv2.data[ index2];
				if( val1 + val2 != 0){
				    sum += (val1 - val2) * (val1 - val2) / ( val1 + val2);
				}
			    index1++;
			    index2++;

		    } else {
			    if( this->IDs[ index1] > fv2.IDs[ index2]){
				    sum += fv2.data[ index2];
				    index2++;
			    } else {
				    sum += this->data[ index1];
				    index1++;
			    }
		    }			
	    }
    }

	if( sum == 0){
		return 1e-30f;
	} else {
	    return sum;    
	}
}


float TFeatureVector::distL1( const TFeatureVector &fv2) const 
{
    float sum = 0;

    if( dense){

        assert( fv2.dense);

        for( int i = 0; i < dimension; i++){
		    sum += fabsf( this->data[ i] - fv2.data[ i]);
        }

    } else {

        assert( ! fv2.dense);

	    int index1 = 0;
	    int index2 = 0;
        int size1 = this->IDLength - 1;
        int size2 = fv2.IDLength - 1;

	    while( index1 < size1 || index2 < size2){

		    if( this->IDs[ index1] == fv2.IDs[ index2]){

    		    sum += fabsf( this->data[ index1++] - fv2.data[ index2++]);

		    } else {
			    if( this->IDs[ index1] > fv2.IDs[ index2]){
				    sum += fabsf( fv2.data[ index2++]);
			    } else {
				    sum += fabsf( this->data[ index1++]);
			    }
		    }			
	    }
    }

	if( sum == 0){
		return 1e-30f;
	} else {
	    return sum;    
	}
}

float TFeatureVector::distL2( const TFeatureVector &fv2) const
{
    float sum = 0;

    if( dense){

        assert( fv2.dense);

        for( int i = 0; i < dimension; i++){
		    sum += ( this->data[ i] - fv2.data[ i]) * ( this->data[ i] - fv2.data[ i]);
        }

    } else {

        assert( ! fv2.dense);

	    int index1 = 0;
	    int index2 = 0;
        int size1 = this->IDLength - 1;
        int size2 = fv2.IDLength - 1;

	    while( index1 < size1 || index2 < size2){

		    if( this->IDs[ index1] == fv2.IDs[ index2]){

    		    sum += ( this->data[ index1] - fv2.data[ index2]) * ( this->data[ index1] - fv2.data[ index2]);
                index1++;
                index2++;

		    } else {
			    if( this->IDs[ index1] > fv2.IDs[ index2]){
				    sum += fv2.data[ index2] * fv2.data[ index2];
                    index2++;
			    } else {
				    sum += this->data[ index1++] * this->data[ index1++];
                    index1++;
			    }
		    }			
	    }
    }

	if( sum == 0){
		return 1e-30f;
	} else {
	    return sqrtf( sum);    
	}
}

float TFeatureVector::distLn( const TFeatureVector &fv2, const int power) const
{
	if( power % 2 != 0){
		cerr << "Error: power in Ln distance must be divisible by 2." << endl;
		return 0;
	}

    float sum = 0;

    if( dense){

        assert( fv2.dense);

        for( int i = 0; i < dimension; i++){
		    sum += powf( this->data[ i] - fv2.data[ i], (float) power);
        }

    } else {

        assert( ! fv2.dense);

	    int index1 = 0;
	    int index2 = 0;
        int size1 = this->IDLength - 1;
        int size2 = fv2.IDLength - 1;

	    while( index1 < size1 || index2 < size2){

		    if( this->IDs[ index1] == fv2.IDs[ index2]){

    		    sum += powf( this->data[ index1++] - fv2.data[ index2++], (float) power);

		    } else {
			    if( this->IDs[ index1] > fv2.IDs[ index2]){
				    sum += powf( fv2.data[ index2++], (float) power);
			    } else {
				    sum += powf( this->data[ index1++], (float) power);
			    }
		    }			
	    }
    }


	if( sum == 0){
		return 1e-30f;
	} else {
	    return powf( sum, 1.0f / (float) power);
	}
}

float TFeatureVector::dotProduct( const TFeatureVector &fv2) const
{

    float sum = 0;

    if( dense){

        assert( fv2.dense);

        for( int i = 0; i < dimension; i++){
		    sum += this->data[ i] * fv2.data[ i];
        }

    } else {

        assert( ! fv2.dense);

	    int index1 = 0;
	    int index2 = 0;
        int size1 = this->IDLength - 1;
        int size2 = fv2.IDLength - 1;

	    while( index1 < size1 || index2 < size2){

		    if( this->IDs[ index1] == fv2.IDs[ index2]){

    		    sum += this->data[ index1++] * fv2.data[ index2++];

		    } else {
			    if( this->IDs[ index1] > fv2.IDs[ index2]){
				    index2++;
			    } else {
				    index1++;
			    }
		    }			
	    }
    }

	if( sum == 0){
		return 1e-30f;
	} else {
	    return sum;
	}
}

float TFeatureVector::HI( const TFeatureVector &fv2) const
{

    float sum = 0;

    if( dense){

        assert( fv2.dense);

        for( int i = 0; i < dimension; i++){
			if( this->data[ i] + fv2.data[ i] != 0){
				sum += min( this->data[ i], fv2.data[ i]) / ( this->data[ i] + fv2.data[ i]);
			}
        }

    } else {

        assert( ! fv2.dense);

	    int index1 = 0;
	    int index2 = 0;
        int size1 = this->IDLength - 1;
        int size2 = fv2.IDLength - 1;

	    while( index1 < size1 || index2 < size2){

		    if( this->IDs[ index1] == fv2.IDs[ index2]){

			    const float val1 = this->data[ index1++];
			    const float val2 = fv2.data[ index2++];
			    if( val1 != 0 && val2 != 0){
				    sum += min( val1, val2) / ( val1 + val2);
			    }

		    } else {
			    if( this->IDs[ index1] > fv2.IDs[ index2]){
				    index2++;
			    } else {
				    index1++;
			    }
		    }			
	    }
    }

	if( sum == 0){
		return 1e-30f;
	} else {
	    return sum;
	}
}



/** Destructors.*/
TFeatureVector::~TFeatureVector(){
    delete[] IDs;
    delete[] data;
}


}
