# OpenVINO people tracker

Simple PoC capable of detect people in a scene taken by cam or by a previously recordered video.  
Mainly used to test the succesful installation of OpenVINO toolkit.

## Install & run

Python3 and virtualenv packages must be installed on your system.

``` bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py <device> [source device]
```

This will execute the pople counter application on your PC.  
The `<device>` argument must be replace with `CPU` or `GPU`, depending on which device you want to perform the inference.

By default the `/dev/video0` stream will be used. If you want to use a different source, let's specify it as a second argument.


## Docker image

You can use the `dockerdima/clea_ai_base` docker image to test the application or to run other applications.  
To do that, create a container with the following command:

```
docker run --rm -it --privileged --net=host -v /tmp/.X11-unix:/tmp/.X11-unix \
            -e DISPLAY=$DISPLAY --device /dev/dri --device /dev/video0:/dev/video0 \
            dockerdima/clea_ai_base
```

and type

```
cd ws/people-tracker-OpenVINO/
source venv/bin/activate
python main.py GPU
```

In case of problems related to X11 server, type  
`xhost +`  
on a different console to accept incoming connections from any client.