parent_path=$( cd "$(dirname "${BASH_SOURCE}")" ; pwd -P )

export PYTHONPATH="$PYTHONPATH:$parent_path/lib/opencv-2.4.11/lib/python2.7/site-packages"
#echo $PYTHONPATH
export LD_LIBRARY_PATH=$parent_path/lib/opencv-2.4.11/lib:$parent_path/lib/2.5.0/lib/:$LD_LIBRARY_PATH
#echo $LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$parent_path/lib/opencv-2.4.11/lib/pkgconfig/:$parent_path/lib/ffmpeg_built/lib/pkgconfig:$PKG_CONFIG_PATH
#

echo $PKG_CONFIG_PATH

