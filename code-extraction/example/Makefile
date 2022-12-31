CC = gcc
CFLAGS = -I.
DEPS = add_one.h
OBJ = main.o add_one.o
BIN = a.out

.PHONY: all install clean

all: install

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

install: $(OBJ)
	$(CC) -o $(BIN) $^ $(CFLAGS)

clean:
	rm -rf *.ll *.o a.out bitcode.* *.bc infer-out/ out/ workspace/
