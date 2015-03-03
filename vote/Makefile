CC = gcc
CFlags = -Wall -lfann -o
CLEAN = combine nets eval *.net 

all:	combine eval nets

combine:
	$(CC) $@.c $(CFlags) $@
eval:
	$(CC) $@.c $(CFlags) $@

clean:
	$(RM) $(CLEAN)

cn:
	$(RM) *.net
