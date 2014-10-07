/*
 * OpenCVFE.cpp
 *
 *  Created on: Sep 4, 2014
 *      Author: ireznice
 */

#include "OpenCVFE.h"
#include <iostream>
#include <fstream>
using namespace std;

//-----------------------------------------------------------------------------

OpenCVFE::OpenCVFE()
{
	cv::initModule_nonfree();
}

//-----------------------------------------------------------------------------

OpenCVFE::~OpenCVFE()
{
}

//-----------------------------------------------------------------------------

void OpenCVFE::Process(protointerface::WorkRequest &request_response)
{
	if (request_response.configuration_size() > 0)
	{
		const protointerface::FEConfiguration &feConf = request_response.configuration(0).fe();
		//do work here

		std::string image = request_response.image();
		vector<unsigned char> input;
		for(unsigned int i = 0; i<image.size(); i++)
		{
			input.push_back(image[i]);
		}

		cv::Mat inputImage = cv::imdecode(input, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Ptr<cv::FeatureDetector> df = cv::FeatureDetector::create("SURF");
		vector<cv::KeyPoint> keypoints;

		try
		{
			df->detect(inputImage, keypoints);
			cv::Ptr<cv::DescriptorExtractor> de = cv::DescriptorExtractor::create("SURF");
			cv::Mat descriptors;
			de->compute(inputImage, keypoints, descriptors);

			//allocate buffer where to store the data;
			//ofstream of("/home/ireznice/OCVFEout", std::ofstream::out);

			int outputsize=descriptors.rows * (descriptors.cols + 6) * sizeof(float);
			float *buf = (float *)malloc(outputsize);
			float *origin = buf;
			int i = 0;
			for(vector<cv::KeyPoint>::const_iterator ki=keypoints.begin(); ki != keypoints.end(); ++i, ++ki)
			{
				//keypoint values
				*buf++ = (float)ki->pt.x;
				*buf++ = (float)ki->pt.y;
				*buf++ = (float)ki->size;
				*buf++ = (float)ki->angle;
				*buf++ = (float)ki->response;
				*buf++ = (float)ki->octave;

				//of<< "0";
				//feature values
				for (int fi=0; fi<descriptors.cols; fi++)
				{
					*buf++ = descriptors.at<float>(i,fi);
					//of << " " << (fi+1) <<":"<< descriptors.at<float>(i,fi);
				}
				//of << endl;
			}
			//of.close();

			protointerface::BlobFloat blobfloat;
			int dim = descriptors.cols + 6;
			memcpy(blobfloat.mutable_dim()->Add(), &dim, sizeof(int));
			dim = descriptors.rows;
			memcpy(blobfloat.mutable_dim()->Add(), &dim, sizeof(int));
			blobfloat.set_data(origin,outputsize);
			request_response.mutable_blob()->CopyFrom(blobfloat);
			free(origin);
		}
		catch(exception &ex)
		{
			cerr << "err";
		}
		cerr << "OpenCVFE processed some data\n";


		//this should be done every-time the FE processing is finished, it reduces the size of further messages
		request_response.clear_image();
	}else
	{
		Log::Message(request_response, "Error while obtaining the worker configuration");
	}
}

//-----------------------------------------------------------------------------
