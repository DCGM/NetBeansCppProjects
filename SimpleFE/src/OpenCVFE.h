/*
 * OpenCVFE.h
 *
 *  Created on: Sep 4, 2014
 *      Author: ireznice
 */

#ifndef OPENCVFE_H_
#define OPENCVFE_H_

#include <interfaces.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class OpenCVFE: public IWorker
{
public:
	OpenCVFE();
	virtual ~OpenCVFE();
	void Process(protointerface::WorkRequest &request_response);
};

#endif /* OPENCVFE_H_ */
