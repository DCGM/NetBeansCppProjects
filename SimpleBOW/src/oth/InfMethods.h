#ifndef __InfMethods_h__
#define __InfMethods_h__


#include <string>
#include <vector>
#include <opencv/cxcore.h>


void LDP( const std::vector< float *>& data, const int dimension, const std::vector< int> &labels, CvMat * projections, CvMat * eigenValues, const int samplesToEstimateCovariance, const bool normDifferences = false, const bool myCovariance = true, const int classesToUse = 0);
void PCA( const std::vector< float *>& data, const int dimension, CvMat * &projections, CvMat * eigenValues, const int samplesToEstimateCovariance);

class TInformaionComputer{

	enum TMethod { UNKNOWN, Sc, MutualInformation, Entropy, Balanced, Mean, ScWithError};

	TMethod method;

public:

	struct TProjectedPoint{
		int label;
		float val;

		static bool cmp( const TProjectedPoint p1, const TProjectedPoint p2){
			return p1.val < p2.val;
		}
	};

	TInformaionComputer( const std::string _method);
	TInformaionComputer( )
		: method( Sc)
	{}

	void getBestThreshold( std::vector< TProjectedPoint> &data, float &threshold, double &score) const;
 	double evaluateThreshold( std::vector< TProjectedPoint> &data, const float &threshold) const;

};

inline void myCalcCovar( CvMat *samplesForCovariance, CvMat * Cs)
{
	cvSetZero( Cs);
	CvMat *row = cvCreateMatHeader( 1, samplesForCovariance->cols, samplesForCovariance->type);

	for( int i = 0; i < samplesForCovariance->rows; i++){
		
		cvGetRow( samplesForCovariance, row, i);
		cvGEMM( row, row, 1.0, Cs, 1.0, Cs, CV_GEMM_A_T);
	}

	cvScale( Cs, Cs, 1.0 / samplesForCovariance->rows);
}

inline double L1Distance( const float * f1, const float * f2, const int dim) 
{
	double sum = 0;
	for( int i = 0; i < dim; i++) {
		sum += fabs( f1[ i] - f2[ i]);
	}
	return sum;
}

inline double L1Distance( const float * f1, const float * f2, const int dim, const double max_distance) 
{
	double sum = 0;
	for( int i = 0; i < dim; i++) {
		sum += fabs( f1[ i] - f2[ i]);
		if( sum > max_distance){
		  return max_distance + 1.0;
		}
	}
	return sum;
}
/*inline double L2Distance( const float * f1, const float * f2, const int dim)
{
	double sum = 0;
	for( int i = 0; i < dim; i++) {
		sum += (f1[ i] - f2[ i]) * (f1[ i] - f2[ i]);
	}
	return sum;
}*/

/*#include <pmmintrin.h>
#include <smmintrin.h>*/

inline double L2Distance( const float * f1, const float * f2, const int dim)
{

/*	__m128 sum = _mm_setzero_ps(); 
	for( int i = 0; i < dim; i += 4){
		const __m128 p1 = _mm_sub_ps( _mm_load_ps( f1 + i), _mm_load_ps( f2 + i));
		sum = _mm_add_ss( sum , _mm_dp_ps( p1, p1, ~(unsigned int)0));
	}
	return _mm_cvtss_f32( sum);*/

	double sum = 0;
	for( int i = 0; i < dim; i++) {
		sum += (f1[ i] - f2[ i]) * (f1[ i] - f2[ i]);
	}
	return sum;
}

inline double L2Distance( const float * f1, const float * f2, const int dim, const double max_distance)
{
	double sum = 0;
	for( int i = 0; i < dim; i++) {
		sum += ( f1[ i] - f2[ i]) * ( f1[ i] - f2[ i]);
		if( sum > max_distance){
			return max_distance + 1;
		}
	}
	return sum;
}

inline double LRDistance( const float * f1, const float * f2, const int dim, const float power)
{
	double sum = 0;
	for( int i = 0; i < dim; i++) {
		sum += powf( fabs( f1[ i] - f2[ i]), power);
	}
	return sum;
}

inline double LRDistance( const float * f1, const float * f2, const int dim, const double max_distance, const float power)
{
	double sum = 0;
	for( int i = 0; i < dim; i++) {
		sum += powf( fabs( f1[ i] - f2[ i]), power);
		if( sum > max_distance){
			return max_distance + 1;
		}
	}
	return sum;
}

inline double X2Distance( const float * f1, const float * f2, const int dim, const float power)
{
	double sum = 0;
	for( int i = 0; i < dim; i++) {
		sum += ((f1[ i] - f2[ i]) * (f1[ i] - f2[ i])) / (f1[ i] + f2[ i]);
	}
	return sum;
}

inline double X2Distance( const float * f1, const float * f2, const int dim, const double max_distance, const float power)
{
	double sum = 0;
	for( int i = 0; i < dim; i++) {
		sum += ((f1[ i] - f2[ i]) * (f1[ i] - f2[ i])) / (f1[ i] + f2[ i]);
		if( sum > max_distance){
			return max_distance + 1;
		}
	}
	return sum;
}

#endif
