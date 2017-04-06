hello: hello.c
	gcc -o hello.out -Wall -x c hello.c

.PHONY: clean

clean:
ifneq "$(wildcard hello.out)" ""
	rm hello.out
endif
