TARGETS = msconnect msclose mslisten msaccept mssend msrecv \
	msrecvraw mssendraw

MATDIR = /room/opt/matlab71

ARCH = $(shell uname -p)

ifeq ($(ARCH),x86_64)
SUFFIX = mexa64
endif

ifeq ($(ARCH),i686)
SUFFIX = mexglx
endif

ifeq ($(ARCH),i586)
SUFFIX = mexglx
endif

ifeq ($(ARCH),i386)
SUFFIX = mexglx
endif

CXX = g++
CC = gcc
CFLAGS = -O3 -DMATLAB_MEX_FILE -fPIC -DMATLAB_MEX_FILE -fomit-frame-pointer -Wall


INCDIR = -I$(MATDIR)/extern/include -I. -I/usr/include

all : $(patsubst %,../%.$(SUFFIX),$(TARGETS))

../msrecv.$(SUFFIX) : msrecv.o matvar.o
	$(CXX) $(CFLAGS) -shared $^ -o $@

../mssend.$(SUFFIX) : mssend.o matvar.o
	$(CXX) $(CFLAGS) -shared $^ -o $@

../%.$(SUFFIX) : %.o
	$(CC) $(CFLAGS) -shared $^ -o $@

%.o : %.cpp
	$(CXX) $(CFLAGS) $(INCDIR) -c $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

clean : 
	rm -rf $(patsubst %,../%.$(SUFFIX),$(TARGETS)) *.o
