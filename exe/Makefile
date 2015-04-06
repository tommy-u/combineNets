CC = gcc
Flags = -Wall -lfann -o
CLEAN = combine eval_bits eval_samples committee_average eval_n_in_m_out *.net 

all:	combine eval_bits eval_samples committee_average eval_n_in_m_out

%: %.c
	$(CC) $^ $(Flags) $@
clean:
	$(RM) $(CLEAN)
cn:
	$(RM) *.net
