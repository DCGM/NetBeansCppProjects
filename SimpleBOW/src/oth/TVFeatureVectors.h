#ifndef __TVFeatureVectors_h__
#define __TVFeatureVectors_h__

#include <string>
#include <vector>
#include <list>
#include <fstream>
#include <climits>
#include <map>
#include <cstdlib>


//#define SPECIFIC_LF

#ifdef WIN32
#define int32_t int
#endif

namespace TRECVID{


/** Structure for single feature vector. Supports both dense and sparse representation.*/
struct TFeatureVector
{
    /** Is true if the vector is stored in dense representation. */
	bool dense;
    /** Contains the values of the feature vector. Can be dense or contain only non-zero elements.*/
	float *data;
    /** If the sparse representation is used, IDs contains pointer to indices of the non-zero elements. The last element of IDs contains INT_MAX.*/
	int32_t *IDs;
    /** Dimension of the feature vector. If dense format - length of data equals to dimension.*/
	int32_t dimension;
    /** Only for sparse representation in which case it is equal to the length of data and the length of IDs is IDLength + 1.*/
	int32_t IDLength;

    /** Read the vector from a stream.*/
	TFeatureVector( std::ifstream *inStream, const bool data8b = false);

    /** Create uninitialized dense vector.*/
	TFeatureVector( const int32_t dimension);

    /** Copy constructor*/
    TFeatureVector( const TFeatureVector& src);

    /** operator= */
    TFeatureVector & operator=( const TFeatureVector& src);

    /** Destructor.*/
    ~TFeatureVector();

    /** Converts the vector to optimal representation (sparse/dense).*/
	void optimize();

    /** Converts the vector to dense representation.*/
	void toDense();

    /** Converts the vector to sparse representation.*/
	void toSparse();

    /** Writes the vector to a binary stream.*/
	void write( std::ofstream *inStream, const bool data8b = false) const;

	void remDim( const std::vector< int> &removeDim);
	
	void scaleDim( const std::vector< std::pair< int, float> > &scales);


    /** Dense vector constructor.*/
    template< class _T>
    TFeatureVector( const _T _data, const int length)
	    : dense( true), data( new float[ length]), IDs( NULL), dimension( length), IDLength( 0)
    {
	    for( int i = 0; i < length; i++){
		    data[ i] = _data[ i];
	    }
    }

    /** Sparse vector constructor.*/
    template< class _T1, class _T2>
    TFeatureVector( const _T1 _data, const _T2 _IDs, const int length, const int _dimension)
	    : dense( false), data( new float[ length]), IDs( new int32_t[ length + 1]), dimension( _dimension), IDLength( length)
    { 
	    for( int i = 0; i < IDLength; i++){
		    data[ i] = _data[ i];
		    IDs[ i]  = _IDs[ i];
	    }
	    IDs[ length] = INT_MAX;
    }

    TFeatureVector & concatenate( const TFeatureVector & fv2);

    void normalizeL1( const float length = 1.0);
    void normalizeL2( const float length = 1.0);
    void normalizeStdDev( const float stdDev = 1.0);

    float distX2( const TFeatureVector &fv2) const;
    float distL1( const TFeatureVector &fv2) const;
    float distL2( const TFeatureVector &hist2) const;
    float distLn( const TFeatureVector &fv2, const int power) const;
    float dotProduct( const TFeatureVector &fv2) const;
    float HI( const TFeatureVector &tv2) const;

}; // TFeatureVector




/** Handles writing feature vectors to file and reading them back. Supports random reading of the feature vectors by using index which is stored at the beginning of the file.*/
class TFeatureVectorFile
{
    /** Is written at the beginning of the feature vector file.*/
	static const std::string fileHead;

    /** Default maximum number of vectors which can be stored in the file - the size of index.*/
    static const long long defMaxSize = 100000;

    /** Name of the accessed file.*/
	std::string filename;
    /** Stream for input operations.*/
	std::ifstream *inStream;
    /** Stream for output operations.*/
	std::ofstream *outStream;
    /** Current position in the index. Is used while sequentially adding new feature vectors in to the file.*/
	long long indexPos;
    /** The index of position of feature vectors in the file.*/
	std::vector< long long> index;

    /** Writes single vector into the file.*/
	static void writeVector( std::ofstream *outStream, const TFeatureVector *fv);
    /** Reads single vector from the file.*/
	static void readVector( std::ifstream *inStream, TFeatureVector *&fv);


public:
	
    /** Opens a file for reading or writing and prepares it for corresponding operations. maxSize sets the maximum number of feature vectors which can be stored in the file when opening for writing.*/
	TFeatureVectorFile( const std::string filename, const bool forWriting, long long maxSize = 0);

    /** Writes the vectors in a file. Some elements of vectors can be set to NULL in which case minimum size empty vector is written in the file.*/
	static void  writeToFile( const std::string filename, const std::vector< TFeatureVector *> &vetors);
    /** Reads all vectors from a file. If the file contains empty feature vectors, vectors will contain NULL elements.*/
	static void  readFromFile( const std::string filename, std::vector< TFeatureVector *> &vetors);

    /** Write several feature vector into the file. The vectors are written dirrectly and are not stored in memory.*/
	void add( const std::vector< TFeatureVector *> &vetors);
    /** Write single feature vector into the file. The vector is written dirrectly and is not stored in memory.*/
	void add( const TFeatureVector * vetor);

    /** Read all vectors from the fille. If the file contains empty feature vectors, vectors will contain NULL elements.*/
	void readAll( std::vector< TFeatureVector *> &vectors);
    /** Read only selected vectors from the fille. Vectors can contain NULL elements after the function is executed.*/
	void read( std::vector< TFeatureVector *> &vectors, const std::vector< long long>& IDsToRead);
    /** Read single feature vector from the file. If the vector is empty in the file, the function returns NULL.*/
	void read( TFeatureVector * &fv, const long long IDToRead, const bool checkPresence = false);

    long long getVectorCount(); 

    /** Writes index if a file is opened for output and deallocates memory.*/
	~TFeatureVectorFile();

}; // class TFeatureVectorFile


// create translation between string and normalization type
enum TNormalization { NO_NORMALIZATION, L1_NORMALIZATION, L2_NORMALIZATION};
extern std::map< std::string, TNormalization> normalizationTranslation;
extern std::map< TNormalization, std::string> invNormalizationTranslation;

enum TKernel { K_X2, K_LINEAR, K_L1, K_L2};
extern std::map< std::string, TKernel> kernelTranslation;
extern std::map< TKernel, std::string> invKernelTranslation;

void prepareTranslations();
void normalize( TFeatureVector * featureVector, const TNormalization normalizationType);

} // namespace TRECVID


#endif

