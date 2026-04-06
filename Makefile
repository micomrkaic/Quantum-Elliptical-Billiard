# Makefile -- Schrodinger elliptic billiard
#
# Linux:   make              (NTHREADS=4)
#          make NTHREADS=8
#   Requires: sudo apt install libsdl2-dev liblapacke-dev liblapack-dev libblas-dev libgsl-dev
#
# macOS:   make macos NTHREADS=10
#   Requires: brew install sdl2 openblas gsl

CC       = gcc
CSTD     = -std=c17
WARN     = -Wall -Wextra
OPT      = -O2
NTHREADS ?= 4

# ---- Linux -------------------------------------------------------
SDL_CF  := $(shell sdl2-config --cflags)
SDL_LF  := $(shell sdl2-config --libs)

CFLAGS  = $(CSTD) $(WARN) $(OPT) -march=native \
          -DNTHREADS=$(NTHREADS) \
          -I/usr/include \
          $(SDL_CF)
LDFLAGS = $(SDL_LF) -llapacke -llapack -lblas -lgsl -lgslcblas -lm -lpthread

TARGET = schrodinger

$(TARGET): schrodinger.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	@echo "Built $(TARGET)  NTHREADS=$(NTHREADS)"

# ---- macOS / Homebrew --------------------------------------------
BREW_SDL := $(shell brew --prefix sdl2     2>/dev/null || echo /opt/homebrew/opt/sdl2)
BREW_OB  := $(shell brew --prefix openblas 2>/dev/null || echo /opt/homebrew/opt/openblas)
BREW_GSL := $(shell brew --prefix gsl      2>/dev/null || echo /opt/homebrew/opt/gsl)

MACOS_CF = $(CSTD) $(WARN) $(OPT) -DNTHREADS=$(NTHREADS) \
           -I$(BREW_SDL)/include/SDL2 \
           -I$(BREW_OB)/include \
           -I$(BREW_GSL)/include
MACOS_LF = -L$(BREW_SDL)/lib -lSDL2 \
           -L$(BREW_OB)/lib  -lopenblas \
           -L$(BREW_GSL)/lib -lgsl -lgslcblas \
           -lm -lpthread

macos: schrodinger.c
	$(CC) $(MACOS_CF) $< -o $(TARGET) $(MACOS_LF)
	@echo "Built $(TARGET) [macOS]  NTHREADS=$(NTHREADS)"

clean:
	rm -f $(TARGET)

.PHONY: clean macos
