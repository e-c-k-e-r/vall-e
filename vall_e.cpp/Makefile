ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

CXX 					= g++

INCS 					+= -I./include
LIBS 					+= -L./lib
	
LINKS 					+= -lggml -lggml-base -lllama -lencodec -lespeak-ng
FLAGS 					+= -march=native -O3 -DVALL_E_EXPORTS

SRCS 					:= $(shell find ./ -name "*.cpp")
OBJS 					+= $(patsubst %.cpp,%.o,$(SRCS))

TARGET 					= vall_e
TARGET_LIB 				= lib$(TARGET).so
TARGET_HEADER 			= $(TARGET).h


%.o: %.cpp
	$(CXX) $(FLAGS) $(INCS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(FLAGS) $(OBJS) $(LIBS) $(INCS) $(LINKS) -o $(TARGET)

$(TARGET_LIB): $(OBJS) 
	$(CXX) $(FLAGS) $(OBJS) $(LIBS) $(INCS) $(LINKS) -o $(TARGET_LIB)

all: $(TARGET_LIB) $(TARGET)

lib: $(TARGET_LIB)

install:
	cp $(TARGET) $(PREFIX)/bin/$(TARGET)
	-cp $(TARGET_LIB) $(PREFIX)/lib/$(TARGET_LIB)
	cp $(TARGET_HEADER) $(PREFIX)/include/$(TARGET_HEADER)

clean:
	@-rm -f $(OBJS)
	@-rm -f $(TARGET)
	@-rm -f $(TARGET_LIB)