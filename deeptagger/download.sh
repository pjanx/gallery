#!/bin/sh -e
# Requirements: Python ~ 3.11, curl, unzip, git-lfs, awk
#
# This script downloads a bunch of models into the models/ directory,
# after any necessary transformations to run them using the deeptagger binary.
#
# Once it succeeds, feel free to remove everything but *.{model,tags,onnx}
git lfs install
mkdir -p models
cd models

# Create a virtual environment for model conversion.
#
# If any of the Python stuff fails,
# retry from within a Conda environment with a different version of Python.
export VIRTUAL_ENV=$(pwd)/venv
export TF_ENABLE_ONEDNN_OPTS=0
if ! [ -f "$VIRTUAL_ENV/ready" ]
then
	python3 -m venv "$VIRTUAL_ENV"
	#"$VIRTUAL_ENV/bin/pip3" install tensorflow[and-cuda]
	"$VIRTUAL_ENV/bin/pip3" install tf2onnx 'deepdanbooru[tensorflow]'
	touch "$VIRTUAL_ENV/ready"
fi

status() {
	echo "$(tput bold)-- $*$(tput sgr0)"
}

# Using the deepdanbooru package makes it possible to use other models
# trained with the project.
deepdanbooru() {
	local name=$1 url=$2
	status "$name"

	local basename=$(basename "$url")
	if ! [ -e "$basename" ]
	then curl -LO "$url"
	fi

	local modelname=${basename%%.*}
	if ! [ -d "$modelname" ]
	then unzip -d "$modelname" "$basename"
	fi

	if ! [ -e "$modelname.tags" ]
	then ln "$modelname/tags.txt" "$modelname.tags"
	fi

	if ! [ -d "$modelname.saved" ]
	then "$VIRTUAL_ENV/bin/python3" - "$modelname" "$modelname.saved" <<-'END'
		import sys
		import deepdanbooru.project as ddp
		model = ddp.load_model_from_project(
			project_path=sys.argv[1], compile_model=False)
		model.export(sys.argv[2])
	END
	fi

	if ! [ -e "$modelname.onnx" ]
	then "$VIRTUAL_ENV/bin/python3" -m tf2onnx.convert \
		--saved-model "$modelname.saved" --output "$modelname.onnx"
	fi

	cat > "$modelname.model" <<-END
		name=$name
		shape=nhwc
		channels=rgb
		normalize=true
		pad=edge
	END
}

# ONNX preconversions don't have a symbolic first dimension, thus doing our own.
wd14() {
	local name=$1 repository=$2
	status "$name"

	local modelname=$(basename "$repository")
	if ! [ -d "$modelname" ]
	then git clone "https://huggingface.co/$repository"
	fi

	# Though link the original export as well.
	if ! [ -e "$modelname.onnx" ]
	then ln "$modelname/model.onnx" "$modelname.onnx"
	fi

	if ! [ -e "$modelname.tags" ]
	then awk -F, 'NR > 1 { print $2 }' "$modelname/selected_tags.csv" \
		> "$modelname.tags"
	fi

	cat > "$modelname.model" <<-END
		name=$name
		shape=nhwc
		channels=bgr
		normalize=false
		pad=white
	END

	if ! [ -e "batch-$modelname.onnx" ]
	then "$VIRTUAL_ENV/bin/python3" -m tf2onnx.convert \
		--saved-model "$modelname" --output "batch-$modelname.onnx"
	fi

	if ! [ -e "batch-$modelname.tags" ]
	then ln "$modelname.tags" "batch-$modelname.tags"
	fi

	if ! [ -e "batch-$modelname.model" ]
	then ln "$modelname.model" "batch-$modelname.model"
	fi
}

# These models are an undocumented mess, thus using ONNX preconversions.
mldanbooru() {
	local name=$1 basename=$2
	status "$name"

	if ! [ -d ml-danbooru-onnx ]
	then git clone https://huggingface.co/deepghs/ml-danbooru-onnx
	fi

	local modelname=${basename%%.*}
	if ! [ -e "$basename" ]
	then ln "ml-danbooru-onnx/$basename"
	fi

	if ! [ -e "$modelname.tags" ]
	then awk -F, 'NR > 1 { print $1 }' ml-danbooru-onnx/tags.csv \
		> "$modelname.tags"
	fi

	cat > "$modelname.model" <<-END
		name=$name
		shape=nchw
		channels=rgb
		normalize=true
		pad=stretch
		size=640
		interpret=sigmoid
	END
}

status "Downloading models, beware that git-lfs doesn't indicate progress"

deepdanbooru DeepDanbooru \
	'https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip'

#wd14 'WD v1.4 ViT v1'        'SmilingWolf/wd-v1-4-vit-tagger'
wd14 'WD v1.4 ViT v2'        'SmilingWolf/wd-v1-4-vit-tagger-v2'
#wd14 'WD v1.4 ConvNeXT v1'   'SmilingWolf/wd-v1-4-convnext-tagger'
wd14 'WD v1.4 ConvNeXT v2'   'SmilingWolf/wd-v1-4-convnext-tagger-v2'
wd14 'WD v1.4 ConvNeXTV2 v2' 'SmilingWolf/wd-v1-4-convnextv2-tagger-v2'
wd14 'WD v1.4 SwinV2 v2'     'SmilingWolf/wd-v1-4-swinv2-tagger-v2'
wd14 'WD v1.4 MOAT v2'       'SmilingWolf/wd-v1-4-moat-tagger-v2'

# As suggested by author https://github.com/IrisRainbowNeko/ML-Danbooru-webui
mldanbooru 'ML-Danbooru Caformer dec-5-97527' 'ml_caformer_m36_dec-5-97527.onnx'
mldanbooru 'ML-Danbooru TResNet-D 6-30000' 'TResnet-D-FLq_ema_6-30000.onnx'
