CXX 					= g++

INCS 					+= -I./include
LIBS 					+= -L./libs
	
LINKS 					+= -lggml -lggml-base -lllama -lencodec
FLAGS 					+= -march=native -O3

SRCS 					:= $(shell find ./ -name "*.cpp")
OBJS 					+= $(patsubst %.cpp,%.o,$(SRCS))

TARGET 					= vall_e

%.o: %.cpp
	$(CXX) $(FLAGS) $(INCS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(FLAGS) $(OBJS) $(LIBS) $(INCS) $(LINKS) -o $(TARGET)

clean:
	@-rm -f $(OBJS)
	@-rm -f $(TARGET)