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

As it's unsafe for Slack API tokens to be shared publically, get the API token from https://api.slack.com/custom-integrations/legacy-tokens and export as an environment variable via
```
export SLACK_API_TOKEN=[75-char-token]
```

For optimization, the object detection code has been split into two seperate scripts that can be run simulatenously, but should be run in seperate instances. Depending on which instance one is running, do the following:

```
cp /detect_pistol/person-camera-session-one.py .
```
or
```
cp /detect_pistol/person-camera-session-two.py .
```
then, for both:
```
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2017_11_08.tar.gz
```

```
tar -xvf faster_rcnn_resnet101_coco_2017_11_08.tar.gz
```

Select Camera

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


Run session one in its own instance, selecting which camera to use
```
python person-camera-session-one.py RECEPTION_EAST
```

Session two can be ran simultaneously with session one in a seperate instance.
```
python person-camera-session-two.py 
```
