CC = gcc
CFlags = -Wall -lfann -o
CLEAN = ctr nets cte *.net 
ctr:
	$(CC) $@.c $(CFlags) $@
cte:
	$(CC) $@.c $(CFlags) $@
nets:
	$(CC) genNets.c $(CFlags) $@

all:	ctr cte nets

clean:
	$(RM) $(CLEAN)

cn:
	$(RM) *.net