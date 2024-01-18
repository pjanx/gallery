#!/bin/sh -e
parse() {
	awk 'BEGIN {
		OFS = FS = "\t"
	} {
		name = $1
		path = $2
		cpu = $3 != ""
		batch = $4
		time = $5

		if (path ~ "/batch-")
			name = name " (batch)"
		else if (name ~ /^WD / && batch > 1)
			next
	} {
		group = name FS cpu FS batch
		if (lastgroup != group) {
			if (lastgroup)
				print lastgroup, mintime

			lastgroup = group
			mintime = time
		} else {
			if (mintime > time)
				mintime = time
		}
	} END {
		print lastgroup, mintime
	}' "${BENCH_LOG:-bench.out}"
}

cat <<END
GPU inference
~~~~~~~~~~~~~
[cols="<,>,>", options=header]
|===
|Model|Batch size|Time
$(parse | awk -F'\t' 'BEGIN { OFS = "|" }
	!$2 { print "", $1, $3, $4 " s" }' | sort -t'|' -nk4)
|===

CPU inference
~~~~~~~~~~~~~
[cols="<,>,>", options=header]
|===
|Model|Batch size|Time
$(parse | awk -F'\t' 'BEGIN { OFS = "|" }
	$2 { print "", $1, $3, $4 " s" }' | sort -t'|' -nk4)
|===
END
