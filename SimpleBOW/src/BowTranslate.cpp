/*
 * fe.cpp
 *
 *  Created on: Sep 2, 2014
 *      Author: ireznice
 */
#include "BowTranslate.h"
#include <iostream>
#include <fstream>

using namespace std;

//-----------------------------------------------------------------------------

void BowTranslate::Process(protointerface::WorkRequest &request_response)
{
	if (request_response.configuration_size() > 0)
	{
		const protointerface::TranslateConfiguration &trConf = request_response.configuration(0).translate();
		int nc = trConf.neighborcount();

		//do work here
		//ofstream of("/home/ireznice/BOWin", std::ofstream::out);
		//ofstream of2("/home/ireznice/BOWout", std::ofstream::out);

		int size = request_response.blob().data().size();
		float *buf = (float *)request_response.blob().data().data();
		int fv_dim = request_response.blob().dim(0);
		int no_fv = request_response.blob().dim(1);
//		for(int fvi=0;fvi < no_fv;fvi++)
//		{
//			for(int fi=0;fi<fv_dim;fi++)
//			{
//				float f = buf[fvi*fv_dim + fi];
//				of << f << " ";
//			}
//			of << endl;
//		}
//		of.close();


		int outdatasize = no_fv * (6 + this->codebookNeighbors * 2) * sizeof(float);
		float *outdata = (float *)malloc(outdatasize);
		float *origin = outdata;

		vector<float> oR;
		oR.reserve(10000);
		oR.resize(10000);
		for(vector<float>::iterator it=oR.begin(); it != oR.end(); it++)
			*it = 0.0f;

		for(int fvi=0;fvi < no_fv;fvi++)
		{
			codebook->getWord( &buf[fvi*fv_dim + 6], this->codebookNN, this->codebookDistance);

			std::vector<double> response(this->codebookNeighbors);
			double sum = 0.0;
			for ( int ni=0 ; ni < codebookNeighbors ; ni++)
			{
				sum += response[ni] = exp( - this->codebookDistance[ni] * this->codebookDistance[ ni] / max(1e-30, (double) 2 * this->codebookStdDev * this->codebookStdDev));
			}

			sum = max( 1e-30, sum);
			for (int ii=0; ii<6; ii++)
			{
				*outdata++ = buf[fvi*fv_dim + ii];
				//of2 << buf[fvi*fv_dim + ii] << " ";

			}

			for ( int ni=0 ; ni < codebookNeighbors ; ni++)
			{
				*outdata++ = (float)codebookNN[ni]+1; //in case the classes are 1...x000
				*outdata++ = (float)(response[ni]/sum);
				//of2 << (float)codebookNN[ni]+1 << ":" << (float)(response[ni]/sum) << " ";
				oR[codebookNN[ni]] += (float)(response[ni]/sum);
			}
			//of2<<endl;

		}

		//of2 << "0";
		//for(int i=0; i<oR.size(); i++)
		//{
		//	if(oR[i] != 0.0f)
		//		of2 << " " << ((int)i+1) << ":" << oR[i];
		//}
		//of2 << endl;
		//of2.close();

		protointerface::BlobFloat blobfloat;
		int dim = 6 + this->codebookNeighbors * 2; //number of columns
		memcpy(blobfloat.mutable_dim()->Add(), &dim, sizeof(int));
		dim = no_fv;	//number of rows
		memcpy(blobfloat.mutable_dim()->Add(), &dim, sizeof(int));
		dim = this->codebookWordCount;	//additionally the codebook size (may be subsequently needed)
		memcpy(blobfloat.mutable_dim()->Add(), &dim, sizeof(int));

		blobfloat.set_data(origin,outdatasize);
		request_response.mutable_blob()->CopyFrom(blobfloat);
		free(origin);

	}else
	{
		Log::Message(request_response, "Error while obtaining the worker configuration");
		throw;
	}
}

//-----------------------------------------------------------------------------

void BowTranslate::InitBOW(std::string cbfname)
{
    this->codebook = new TProjKmeans( cbfname );
    double codebookAvgWordDist = this->codebook->getAvgWordDistance();
    this->codebookWordCount = this->codebook->getWordCount();
    this->codebookAvgWordDist = codebookAvgWordDist;
    this->codebookDimension = this->codebook->getInputDimension();
    this->codebookStdDev = 1 * codebookAvgWordDist;


    this->codebookNeighbors = 32;

    this->codebookNN.resize(codebookNeighbors);
    this->codebookNN.reserve(codebookNeighbors);
    this->codebookDistance.reserve(codebookNeighbors);
    this->codebookDistance.resize(codebookNeighbors);

}

//-----------------------------------------------------------------------------

BowTranslate::~BowTranslate()
{
	if(this->codebook)
	{
		delete this->codebook;
	}
}

//-----------------------------------------------------------------------------
