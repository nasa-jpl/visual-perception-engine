#!/bin/bash

cd /home/naisr/"$ROS_WORKSPACE"/nn_engine/resources

# cheetah example
example1="cheetah"
example1_url=https://upload.wikimedia.org/wikipedia/commons/6/62/Cheetahs_on_the_Edge_%28Director%27s_Cut%29.ogv

mkdir -p $example1/video
wget -O $example1/video/video.ogv $example1_url

echo "Converting video to frames..."
mkdir -p $example1/frames
python3 vid2frames.py --video_path $example1/video/video.ogv --output_path $example1/frames