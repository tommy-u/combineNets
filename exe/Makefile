CC = gcc
CFlags = -Wall -lfann -o
CLEAN = combine eval_bits eval_samples committee_average *.net 

all:	combine eval_bits eval_samples committee_average

combine:
	$(CC) $@.c $(CFlags) $@
eval_bits:
	$(CC) $@.c $(CFlags) $@
eval_samples:
	$(CC) $@.c $(CFlags) $@
committee_average:
	$(CC) $@.c $(CFlags) $@

clean:
	$(RM) $(CLEAN)
cn:
	$(RM) *.net
