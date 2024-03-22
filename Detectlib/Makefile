CXX = g++
# CXXFLAGS = -std=c++11 -Wall
CXXFLAGS = -Wall
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

SRCS = DetectFake.cpp
TARGET = DetectFake.so

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o  $@ $^ $(OPENCV_CFLAGS) $(OPENCV_LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean

#g++ -shared -o libdetect.so DetectFake.cpp -fPIC `pkg-config --cflags --libs opencv4`
#g++ -shared -o DetectFake.so DetectFake.cpp -fPIC `pkg-config --cflags --libs opencv4`