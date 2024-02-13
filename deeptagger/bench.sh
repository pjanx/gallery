#!/bin/sh -e
if [ $# -lt 2 ] || ! [ -x "$1" ]
then
	echo "Usage: $0 DEEPTAGGER FILE..."
	echo "Run this after using download.sh, from the same directory."
	exit 1
fi

runner=$1
shift
log=bench.out
: >$log

run() {
	opts=$1 batch=$2 model=$3
	shift 3

	for i in $(seq 1 3)
	do
		start=$(date +%s)
		"$runner" $opts -b "$batch" -t 0.75 "$model" "$@" >/dev/null || :
		end=$(date +%s)
		printf '%s\t%s\t%s\t%s\t%s\n' \
			"$name" "$model" "$opts" "$batch" "$((end - start))" | tee -a $log
	done
}

for model in models/*.model
do
	name=$(sed -n 's/^name=//p' "$model")
	for batch in 1 4 16
	do
		run ""    $batch "$model" "$@"
		run --cpu $batch "$model" "$@"
	done
done
