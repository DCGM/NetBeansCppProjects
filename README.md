NetBeansCppProjects
===================

NetBeansProjects

git repository must be cloned in /home/ireznice/somedir
because of the NetBeansIDE stores all paths in a relative way.

the content of AllInOne directory should be working everywhere on medusa, 
when moved elsewhere the LIB directories and include directories in file Makefile.i should be changed accordingly
to build the projects just type:
./make_struct
make

to clean the mess just type: 
make clean


Messages format:

FE --> BOW
----------

BlobFloat object is used, floats array is transfered
BlobFloat.dim contains 2 numbers, first one is the number of columns, second one is the number of rows of transferred array
BlobFloat.data holds the (float *) pointer to data.
each row is in the following format:
PositionX PositionY SIZE ANGLE RESPONSE OCTAVE <1st feature value> <2nd feature value> ... <last feature value> 
In case the size of feature vector is 64, the row contain 6 + 64 = 70 values. 


BOW --> Search
--------------

BlobFloat object is used, floats array is transfered
BlobFloat.dim contains 3 numbers, first one is the number of columns, second one is the number of rows of transferred array, the last one is the dimension of codebook used
BlobFloat.data holds the (float *) pointer to data.
each row is in the following format:
PositionX PositionY SIZE ANGLE RESPONSE OCTAVE <1st codebook index> <1st feature value> <2nd codebook index> <2nd feature value> ... <last codebook index> <last feature value>
In case the number of searched nearest neighbours is 16, the row contain 6 + 2*16 = 38 values.
Codebook indexes are counted starting from number ONE!


Search --> Webserver
--------------------

ResultList object is used for such purposes.

