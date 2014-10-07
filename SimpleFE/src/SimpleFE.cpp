/*
 * fe.cpp
 *
 *  Created on: Sep 2, 2014
 *      Author: ireznice
 */
#include "SimpleFE.h"
#include <iostream>
#include <fstream>

using namespace std;

//-----------------------------------------------------------------------------

void SimpleFE::Process(protointerface::WorkRequest &request_response)
{
	if (request_response.configuration_size() > 0)
	{
		const protointerface::FEConfiguration &feConf = request_response.configuration(0).fe();

		//do work here
		cerr << "simpleFE processed some data\n";
		//put some message
		Log::Message(request_response, "Foo");


		//put some dummy data into response
		// we are outputting 5 float vectors with dimensionality 50;
		protointerface::BlobFloat bf;
		int dim = 50;
		memcpy(bf.mutable_dim()->Add(), &dim, sizeof(int));
		dim = 5;
		memcpy(bf.mutable_dim()->Add(), &dim, sizeof(int));
		float *buf = (float*)malloc(sizeof(float)*50*5);
		for(int i=0;i<50*5;i++)
		{
			buf[i]=(float)i*3.1415926;
		}

		//ofstream of("/home/ireznice/FEout");
		//of.write((char *)buf,sizeof(float)*50*5);
		//of.close();

		bf.set_data(buf, sizeof(float)*50*5);
		request_response.mutable_blob()->CopyFrom(bf);
		free(buf);

		//this should be done every-time the FE processing is finished, it reduces the size of further messages
		request_response.clear_image();
	}else
	{
		Log::Message(request_response, "Error while obtaining the worker configuration");
		throw;
	}
}

//-----------------------------------------------------------------------------
