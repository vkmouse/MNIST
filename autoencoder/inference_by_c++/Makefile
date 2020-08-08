OPENCVPATH = C:\opencv
TARGET = main

OBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
CC = g++
OPTION = -O3 \
-I $(OPENCVPATH)\include \
-L $(OPENCVPATH)\x64\mingw\bin \
-l libopencv_core411 \
-l libopencv_highgui411 \
-l libopencv_imgcodecs411 \
-l libopencv_imgproc411 \
-l libopencv_dnn411 

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(OPTION) -lm 

%.o: %.cpp $(HEADER)
	$(CC) $(CFLAGS) $(OPTION)  -c $<

clean:
	del -rf *.o
	del $(TARGET).exe
