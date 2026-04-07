# Makefile -- Schrodinger elliptic billiard
#
# Usage:
#   Linux:   make [NTHREADS=8]
#   macOS:   make [NTHREADS=10]
#   (platform is auto-detected via uname)
#
# Dependencies:
#   Linux:  sudo apt install libsdl2-dev libgsl-dev
#   macOS:  brew install sdl2 gsl openblas

CC       = gcc
CSTD     = -std=c17
WARN     = -Wall -Wextra
OPT      = -O2
NTHREADS ?= 4
TARGET   = schrodinger

# ── auto-detect platform ──────────────────────────────────────────
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
# ── macOS ─────────────────────────────────────────────────────────
BREW     := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
SDL_PRE  := $(shell brew --prefix sdl2    2>/dev/null || echo $(BREW)/opt/sdl2)
GSL_PRE  := $(shell brew --prefix gsl     2>/dev/null || echo $(BREW)/opt/gsl)
OB_PRE   := $(shell brew --prefix openblas 2>/dev/null || echo $(BREW)/opt/openblas)

CFLAGS   = $(CSTD) $(WARN) $(OPT) -DNTHREADS=$(NTHREADS) \
           -I$(SDL_PRE)/include \
           -I$(GSL_PRE)/include \
           -I$(OB_PRE)/include
LDFLAGS  = -L$(SDL_PRE)/lib  -lSDL2 \
           -L$(GSL_PRE)/lib  -lgsl -lgslcblas \
           -L$(OB_PRE)/lib   -lopenblas \
           -lm -lpthread

else
# ── Linux ─────────────────────────────────────────────────────────
SDL_CF   := $(shell sdl2-config --cflags 2>/dev/null || echo -I/usr/include/SDL2 -D_REENTRANT)
SDL_LF   := $(shell sdl2-config --libs   2>/dev/null || echo -lSDL2)

CFLAGS   = $(CSTD) $(WARN) $(OPT) -march=native -DNTHREADS=$(NTHREADS) \
           -I/usr/include \
           $(SDL_CF)
LDFLAGS  = $(SDL_LF) -lgsl -lgslcblas -lm -lpthread

endif

$(TARGET): schrodinger.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	@echo "Built $(TARGET)  [$(UNAME)]  NTHREADS=$(NTHREADS)"

clean:
	rm -f $(TARGET)

.PHONY: clean
