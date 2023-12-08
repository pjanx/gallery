.POSIX:
.SUFFIXES:

outputs = gallery initialize.go public/mithril.js
all: $(outputs)

gallery: main.go initialize.go
	go build -tags "" -gcflags="all=-N -l" -o $@
initialize.go: initialize.sql gen-initialize.sh
	./gen-initialize.sh initialize.sql > $@
public/mithril.js:
	curl -Lo $@ https://unpkg.com/mithril/mithril.js
clean:
	rm -f $(outputs)
