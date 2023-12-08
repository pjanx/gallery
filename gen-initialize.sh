#!/bin/sh -e
gofmt <<EOF
package ${GOPACKAGE:-main}

const initializeSQL = \`$(sed 's/`/` + "`" + `/g' "$@")\`
EOF
