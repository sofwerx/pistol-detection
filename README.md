# pistol-detection


###

`
nvidia-docker run --rm --network host --privileged -it -v ~/.Xauthority:/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /dev/video0:/dev/video0 -v /home/david/Documents/gundetection/object-detection/:/home/david/Documents/gundetection/object-detection/ -v $HOME/android_tensorflow_gun_detection/tf_files:/tf_files  --device /dev/snd gpu_tf bash

`
