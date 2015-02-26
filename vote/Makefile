CC = gcc
CFlags = -Wall -lfann -o
CLEAN = combine nets eval *.net 
combine:
	$(CC) $@.c $(CFlags) $@
eval:
	$(CC) $@.c $(CFlags) $@
nets:
	$(CC) genNets.c $(CFlags) $@

all:	combine eval nets

clean:
	$(RM) $(CLEAN)

cn:
	$(RM) *.net