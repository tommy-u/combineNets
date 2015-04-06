k-CC = gcc
FLAGS = -Wall -lfann

all: buildNets bootstrap
%: %.c
	$(k-CC) $^ $(FLAGS) -o $@
clean: rm buildNets bootstrap