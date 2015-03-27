CC = gcc
CFlags = -Wall -lfann -o
CLEAN = combine eval_bits eval_samples *.net 

all:	combine eval_bits eval_samples

combine:
	$(CC) $@.c $(CFlags) $@
eval_bits:
	$(CC) $@.c $(CFlags) $@
eval_samples:
	$(CC) $@.c $(CFlags) $@
clean:
	$(RM) $(CLEAN)

cn:
	$(RM) *.net
