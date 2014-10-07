/*
 * fe.cpp
 *
 *  Created on: Sep 2, 2014
 *      Author: ireznice
 */
#include "SimpleSearch.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <functional>
#include <sstream>

using namespace std;

//-----------------------------------------------------------------------------

void SimpleSearch::Process(protointerface::WorkRequest &request_response)
{
	if (request_response.configuration_size() > 0)
	{
		const protointerface::MatchConfiguration &mConf = request_response.configuration(0).match();
		string db = mConf.database();
		int  numClosest = mConf.listsize();

		//do work here
		cerr << "simpleSearch processed some data\n";
		//put some message
		Log::Message(request_response, "Foo");


		//ofstream of("/home/ireznice/Searchin", std::ofstream::out);
		//ofstream of2("/home/ireznice/BOWout", std::ofstream::out);

		int size = request_response.blob().data().size();
		float *buf = (float *)request_response.blob().data().data();
		int fv_dim = request_response.blob().dim(0);
		int no_fv = request_response.blob().dim(1);
		int cbsize = request_response.blob().dim(2);
		/*
		for(int fvi=0;fvi < no_fv;fvi++)
		{
			for(int fi=0;fi<fv_dim;fi++)
			{
				float f = buf[fvi*fv_dim + fi];
				of << f << " ";
			}
			of << endl;
		}
		of.close();
		 */

		std::vector<float> featute_representation;
		featute_representation.resize(cbsize);
		featute_representation.reserve(cbsize);
		for(int i=0; i<featute_representation.size(); i++)
			featute_representation[i]=0.0f;

		for(int fvi=0; fvi < no_fv; fvi++)
			for(int fi=6; fi<fv_dim; fi+=2)
				if((int)buf[fvi*fv_dim + fi]-1 >=0 && (int)buf[fvi*fv_dim + fi]-1<cbsize)
					featute_representation[(int)buf[fvi*fv_dim + fi]-1] += buf[fvi*fv_dim + fi+1];

		vector<pair<int, float > > indexes = this->Search(featute_representation, numClosest);

		//put some output data
		protointerface::ResultList *output_list = request_response.mutable_result();
		for (vector<pair<int, float> >::const_iterator it=indexes.begin(); it!=indexes.end(); it++)
		{
			stringstream ss;
			ss << "http://foo/" << base_descr[it->first];
			output_list->add_url(ss.str());
			output_list->add_score((float)it->second);
		}

	}else
	{
		Log::Message(request_response, "Error while obtaining the worker configuration");
		throw;
	}
}

//-----------------------------------------------------------------------------
void SimpleSearch::Init(string basefile)
{
	ifstream ifile;
    vector<string> hhh;
	ifile.open(basefile.c_str());
	char templine[20000]="";
	string filename;
	while (ifile.good())
	{
		ifile.getline(templine, 20000);
		stringstream ss(templine);
		if(strlen(templine)<2)
		    continue;
		ss >> filename;
		vector<float> temp;
		temp.reserve(1000);
		temp.resize(1000);

		while(ss.good())
		{
			string piece;
			ss >> piece;
			if(piece.size()<1)
				continue;
			size_t pos = piece.find(':');
			if (pos == string::npos)
			{
				//error
			}
			else
			{
				int idx = atoi(piece.substr(0,pos).c_str());
				float val = atof(piece.substr(pos+1).c_str());
				temp[idx-1] = val;
			}
		}

		this->base_descr.push_back(filename);
		hhh.push_back(filename);
		this->base.push_back(temp);
	}
	//base_descr = hhh;
	ifile.close();
	return;

    //fill in fake values
    this->base.reserve(400);
    this->base.resize(400);
    for (int i = 0; i < 400; i++) {
        this->base[i].reserve(1000);
        this->base[i].resize(1000);
        for (int y = 0; y < 1000; y++) {
            this->base[i][y] = 0.0f;
        }

        for (int y = 0; y < 1000; y += 1) {
            this->base[i][y] = rand() % 100000;
        }
    }
}

bool foo(pair<int, float> a, pair<int, float> b)
{
	return a.second < b.second;
}

vector<pair<int, float> > SimpleSearch::Search(vector<float> &in, int nearest)
{
	vector<float> res;
	res.reserve(this->base.size());
	res.resize(this->base.size());
	for(int g=0;g<this->base.size(); g++)
	{
		for(int si=0;si<in.size();si++)
		{
			if(0.0f != (float)(in[si]) && 0.0f != this->base[g][si])
			{
				res[g] =  ((float)(in[si]) - (float)(this->base[g][si]))*((float)(in[si]) - (float)(this->base[g][si]));
			}
		}
		res[g] = sqrt(res[g]);
	}

	std::vector<pair<int, float> > pairs;
	for(int ri=0;ri<res.size();ri++)
	{
		pair<int, float> p;
		p.first = ri;
		p.second = res[ri];
		pairs.push_back(p);
	}

	sort(pairs.begin(), pairs.end(), foo);
	pairs.resize(nearest);
	return pairs;
}



