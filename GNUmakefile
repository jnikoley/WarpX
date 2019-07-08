AMREX_HOME  ?= ../amrex
PICSAR_HOME ?= ../picsar
OPENBC_HOME ?= ../openbc_poisson

DEBUG	= FALSE
#DEBUG	= TRUE

#DIM     = 2
DIM	= 3

COMP = gcc
#COMP = intel
#COMP = pgi

TINY_PROFILE   = TRUE
#PROFILE       = TRUE
#COMM_PROFILE  = TRUE
#TRACE_PROFILE = TRUE

USE_OMP   = TRUE
USE_GPU   = FALSE

EBASE     = main

USE_PYTHON_MAIN = FALSE

USE_SENSEI_INSITU = FALSE
USE_ASCENT_INSITU = FALSE

WarpxBinDir = Bin

USE_PSATD = FALSE
USE_RZ = FALSE

DO_ELECTROSTATIC = FALSE

ifeq ($(USE_RZ),TRUE)
  # DIM is set here so that it is properly set for
  # all of the included files.
  DIM = 2
endif

WARPX_HOME := .
include $(WARPX_HOME)/Source/Make.WarpX
