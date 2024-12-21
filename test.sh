#!/bin/sh -xe
cd "$(dirname "$0")"
make gallery
target=/tmp/G input=/tmp/G/Test
rm -rf $target

mkdir -p $target $input/Test $input/Empty
gen() { magick "$@"; sha1=$(sha1sum "$(eval echo \$\{$#\})" | cut -d' ' -f1); }

gen wizard: $input/wizard.webp
gen -seed 10 -size 256x256 plasma:fractal \
	$input/Test/dhash.jpg
gen -seed 10 -size 256x256 plasma:fractal \
	$input/Test/dhash.png
sha1duplicate=$sha1
cp $input/Test/dhash.png \
	$input/Test/multiple-paths.png

gen -seed 15 -size 256x256 plasma:fractal \
	$input/Test/excluded.png

gen -seed 20 -size 160x128 plasma:fractal \
	-bordercolor transparent -border 64 \
	$input/Test/transparent-wide.png
gen -seed 30 -size 1024x256 plasma:fractal \
	-alpha set -channel A -evaluate multiply 0.2 \
	$input/Test/translucent-superwide.png

gen -size 96x96 -delay 10 -loop 0 \
	-seed 111 plasma:fractal \
	-seed 222 plasma:fractal \
	-seed 333 plasma:fractal \
	-seed 444 plasma:fractal \
	-seed 555 plasma:fractal \
	-seed 666 plasma:fractal \
	$input/Test/animation-small.gif
sha1animated=$sha1
gen $input/Test/animation-small.gif \
	$input/Test/video.mp4

./gallery init $target
./gallery sync -exclude '/excluded[.]' $target $input "$@"
./gallery thumbnail $target
./gallery dhash $target
./gallery tag $target test "Test space" <<-END
	$sha1duplicate	foo	1.0
	$sha1duplicate	bar	0.5
	$sha1animated	foo	0.8
END

# TODO: Test all the various possible sync transitions.
mv $input/Test $input/Plasma
./gallery sync -exclude '/excluded[.]' $target $input

./gallery web $target :8080 &
web=$!
trap "kill $web; wait $web" EXIT INT TERM
sleep 0.25

call() (curl http://localhost:8080/api/$1 -X POST --data-binary @-)

# TODO: Verify that things are how we expect them to be.
echo '{"path":"'"$(basename "$input")"'"}' | call browse
echo '{}' | call tags
echo '{}' | call duplicates
echo '{}' | call orphans
echo '{"sha1":"'"$sha1duplicate"'"}' | call info
echo '{"sha1":"'"$sha1duplicate"'"}' | call similar
