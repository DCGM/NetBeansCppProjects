
CXX =g++

CXXFLAGS = -I/home/ireznice/local/include -I. -I.. -I./src -g `pkg-config --cflags opencv`

LIBS = -lboost_system -lprotobuf -lSimpleAmqpClient -lrabbitmq -lboost_chrono \
`pkg-config --libs opencv` -L/home/ireznice/local/lib -L/home/ireznice/local/lib/x86_64-linux-gnu

