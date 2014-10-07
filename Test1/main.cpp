/* 
 * File:   main.cpp
 * Author: ireznice
 *
 * Created on September 23, 2014, 8:24 AM
 */

#include <cstdlib>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/core.hpp>
#include <boost/lambda/loops.hpp>
#include <list>
#include <iostream>

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) 
{
    list<int> v(10);
    int counter=0;
    for_each(v.begin(), v.end(), boost::lambda::_1 = counter++);
    
    for_each(v.begin(), v.end(), (cout << boost::lambda::_1, std::cout << endl) );
    return 0;
}

