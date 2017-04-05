CUDA_INSTALL_PATH ?= /usr/local/cuda
VER =

CXX := /usr/bin/g++$(VER)
CC := /usr/bin/g++$(VER)
LINK := /usr/bin/g++$(VER) -fPIC
CCPATH := ./gcc$(VER)
NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc -ccbin $(CCPATH)

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Libraries
LIB_CUDA := -lcuda

# Options
NVCCOPTIONS = -arch sm_20 -ptx

#COMPILER_FLAGS specifies the additional compilation options we're using
COMPILER_FLAGS = -std=gnu++11

#LINKER_FLAGS specifies the libraries that we're using
#LINKER_FLAGS = -I/home/z1126133/n-body-simulation/include/SDL2 -D_REENTRANT -L/home/z1126133/n-body-simulation/lib -Wl,-rpath,/home/z1126133/n-body-simulation/lib -Wl,--enable-new-dtags -lSDL2
LINKER_FLAGS = $(shell bin/sdl2-config --cflags --libs)

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)
CXXFLAGS += $(COMMONFLAGS) $(COMPILER_FLAGS)
CFLAGS += $(COMMONFLAGS)

#OBJ_NAME specifies the name of our executable
OBJ_NAME = simulation_binary

#OBJS specifies which files to compile as part of the project
CUDA_OBJS = src/cuda/verlet.ptx
OBJS = src/graphic.cpp.o src/verlet.cpp.o src/main.cpp.o src/findUnion.cpp.o
LINKLINE = $(LINK) -o $(OBJ_NAME) $(OBJS) $(LIB_CUDA) $(LINKER_FLAGS)


.SUFFIXES: .c .cpp .cu .o
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_NAME): prepare $(OBJS) $(CUDA_OBJS)
	$(LINKLINE)

#This is the target that compiles executable
#all : $(OBJS)
#	$(CC) $(OBJS) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $(OBJ_NAME)

clean:
	rm -rf $(OBJ_NAME) src/*.o src/cuda/*.ptx

prepare:
	rm -rf $(CCPATH);\
	mkdir -p $(CCPATH);\
	ln -s $(CXX) $(CCPATH)/g++;\
	ln -s $(CC) $(CCPATH)/gcc;
