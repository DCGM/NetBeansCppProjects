/* 
 * File:   FVFixer.cpp
 * Author: ireznice
 *
 * Created on September 30, 2014, 7:35 PM
 */

#include <cstdlib>
#include "src/TVFeatureVectors.h"
#include <iostream>
#include <math.h>

using namespace std;
using namespace TRECVID;

/*
 * 
 */
int main(int argc, char** argv) 
{
    if(argc < 3)
    {
        cerr << "Usage: app infile outfile\n";
        exit(-1);
    }
    
    TFeatureVectorFile ifile(argv[1],false);
    TFeatureVectorFile ofile(argv[2],true);
    
    int size = ifile.getVectorCount();
    for (int i=0; i<size; i++)
    {
        TFeatureVector *fv = NULL;
        ifile.read(fv, i, false);
        if(fv)
        {
            //cerr << i << " " << fv->dense <<endl;
            
            for(int fi=0; fi<fv->dimension; fi++)
            {
                if(isnan(fv->data[fi]))
                {
                    fv->data[fi] = 0.9;
                }
            }
            
            ofile.add(fv);
        }
        else
        {
            exit(-1);
        }
    }
    
    return 0;
}

