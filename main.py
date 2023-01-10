
import sys
import cv2
import numpy as np
import pandas as pd
import time

from openvino.inference_engine import IECore




class Main :
    def __init__ (self) :
        
        self.ie = IECore()

        pd_model_name = "person-detection-retail-0013"
        pd_model_path = './models/{}/FP32'.format(pd_model_name)
        print ("Loading AI network")
        self.pd_net = self.ie.read_network(
            model="{}/{}.xml".format(pd_model_path, pd_model_name),
            weights="{}/{}.bin".format(pd_model_path, pd_model_name)
        )
        self.pd_executor = self.ie.load_network (self.pd_net, sys.argv[1])
        self.PERSON_DETECTION_TRESHOLD_CONFIDENCE = 0.9

        self.input_info_pd = next(iter(self.pd_executor.input_info))
        self.net_dims = self.pd_net.input_info[self.input_info_pd].tensor_desc.dims

        print ("\n====================\nAll networks loaded!\n====================\n")

        source_stream = "/dev/video0"
        if len(sys.argv) == 3 :
            source_stream = sys.argv[2]
        self.camera = cv2.VideoCapture(source_stream)

        cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)


    def execute_person_detection(self, frame, n, c, h, w, input_info):
        
        # Adapting image to neural network reqs
        blob = cv2.resize(frame, (w, h))  # Resize width & height
        blob = blob.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        blob = np.expand_dims(blob, 0)

        res = self.pd_executor.infer(inputs={input_info: blob})

        resolution_x = frame.shape[1]
        resolution_y = frame.shape[0]
        (real_y, real_x), (resized_y, resized_x) = frame.shape[:2], blob.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

        out_info = []

        if res:
            for i in range(res["detection_out"].shape[2]):
                conf = res["detection_out"][0,0,i,2]
                if conf > self.PERSON_DETECTION_TRESHOLD_CONFIDENCE:
                    print ('+1 with conf {}'.format(conf))

                    (x_min, y_min, x_max, y_max) = [
                        int(max(corner_position * ratio_y *resized_y, 10)) if idx % 2
                        else int(max(corner_position * ratio_x *resized_x, 10))
                        for idx, corner_position in enumerate(res["detection_out"][0,0,i,3:])
                    ]
                    xmin = x_min
                    ymin = y_min
                    xmax = x_max
                    ymax = y_max
                    out_info.append ({'min':{'x':xmin, 'y':ymin}, 'max':{'x':xmax, 'y':ymax}})
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 128, 255), 3)  # drawing rectangle to main image        
        
        return frame, out_info



    def run (self) :
        print ("Running!")

        exit    = False

        while not exit :
            has_frame, frame = self.camera.read()
    
            if frame is None:
                break

            frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
            out, data = self.execute_person_detection (frame.copy(), self.net_dims[0], self.net_dims[1], self.net_dims[2], 
                                                    self.net_dims[3], self.input_info_pd)
            cv2.imshow('Display', out)

            key = cv2.waitKey(10)
            if key == 113 :
                exit = True
        




if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print ("\nWrong arguments number!\n\t\t\tUsage:\tpython main.py <device> [source stream]\n\n")
        raise
    if sys.argv[1] != "CPU" and sys.argv[1] != "GPU":
        print ("\nDevice on which execute inference must be 'CPU' or 'GPU'\n\n")
        raise

    main    = Main()
    main.run()