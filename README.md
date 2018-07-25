# pistol-detection

```
git clone https://github.com/sofwerx/assault-rifle-detection.git $HOME/Documents/pistol-detection
```
```
cd $HOME/Documents/pistol-detection
```

```
docker build -t gpu_tf .
```

```
xhost +local:docker
```

```
nvidia-docker run --rm --network host --privileged -it -v ~/.Xauthority:/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /dev/video0:/dev/video0  -v $HOME/Documents/pistol-detection/tf_files:/tf_files  --device /dev/snd gpu_tf bash
```

```
cd object_detection
```

```
cp /tf_files/person-camera.py .
```

```
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2017_11_08.tar.gz
```

```
tar -xvf faster_rcnn_resnet101_coco_2017_11_08.tar.gz
```

Select Camera and GPU Allocation

RECEPTION_EAST
RECEPTION_WEST
DIRTYWERX_NORTH
DIRTYWERX_SOUTH
THUNDERDRONE_INDOOR_EAST
THUNDERDRONE_INDOOR_WEST
OUTSIDE_WEST
OUTSIDE_NORTH_WEST
OUTSIDE_NORTH
OUTSIDE_NORTH_EAST
DIRTYWERX_RAMP

```
python pistol-detection.py RECEPTION_EAST 100
```
