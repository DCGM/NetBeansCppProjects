/*
 * fe.h
 *
 *  Created on: Sep 2, 2014
 *      Author: ireznice
 */

#ifndef FE_H_
#define FE_H_

#include "interfaces.h"
#include "ProjKMeans.h"
#include "TVFeatureVectors.h"

class BowTranslate : public IWorker
{
	TProjKmeans *codebook;
	double codebookAvgWordDist;
	int codebookWordCount;
	int codebookNeighbors;
	int codebookDimension;
	float codebookStdDev;
    std::vector<int> codebookNN;
    std::vector<double> codebookDistance;

	void Process(protointerface::WorkRequest &request_response);
public:
	void InitBOW(std::string cbfname);
	~BowTranslate();
};

#endif /* FE_H_ */
