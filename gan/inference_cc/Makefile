HEADER = Model.hpp Tensor.hpp
TENSERFLOW_PATH = C:\libtensorflow-cpu-windows-x86_64-1.15.0
OPENCV_PATH = C:\opencv
TARGET = inference

OBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
CC = g++
OPTION = -O3 \
-I $(TENSERFLOW_PATH)\include \
-L $(TENSERFLOW_PATH)\lib \
-I $(OPENCV_PATH)\include \
-L $(OPENCV_PATH)\x64\mingw\lib \
-l tensorflow \
-l libopencv_core411 \
-l libopencv_highgui411 \
-l libopencv_imgcodecs411 \
-l libopencv_imgproc411

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(OPTION) -lm 

%.o: %.cpp $(HEADER)
	$(CC) $(CFLAGS) $(OPTION)  -c $<

clean:
	del -rf *.o
	del $(TARGET).exe
