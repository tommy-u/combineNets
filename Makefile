all:
	cd bagging; make all
	cd exe; make all
clean:
	cd bagging; make clean
	cd exe; make clean