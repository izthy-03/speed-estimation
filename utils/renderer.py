import cv2 as cv
import numpy as np
import threading 
import time
from tqdm import tqdm


class Renderer:
    def __init__(self, input_path, output_path, buf_size_limit_MB=4096):
        """
        Constructor for the Renderer class.

        Args:
        - input_path (str): The path to the input video.
        - output_path (str): The path to the output video.
        - buf_size_limit_MB (int): The maximum buffer size in MB.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.cap = cv.VideoCapture(input_path)

        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.frame_id = 0

        # Externelly registered painting functions
        self.painter_funcs = []

        self.mux = threading.Lock()
        self.writer_buf = [None for _ in range(self.total_frames+1)]
        self.buf_size = 0
        self.buf_size_limit_MB = buf_size_limit_MB

    
    def __renderer(self, worker_id):
        """
        Worker function to render the video
        """
        while True:
            ret, frame = None, None
            curr_frame_id = None

            self.mux.acquire()
            if self.buf_size > self.buf_size_limit_MB * 1024 * 1024:
                self.mux.release()
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            curr_frame_id = self.frame_id
            if not ret:
                self.mux.release()
                break
            # print(f"Worker {worker_id} processing frame {curr_frame_id}...")
            self.frame_id += 1

            self.mux.release()
            
            for func in self.painter_funcs:
                frame = func(frame, curr_frame_id)
            
            with self.mux:
                self.buf_size += frame.nbytes
                self.writer_buf[curr_frame_id] = frame
            

    def __writer(self, output_path):
        """
        Writer function to write the video
        """
        curr_frame_id = 0
        height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        size = (int(width), int(height))
        rate = self.cap.get(cv.CAP_PROP_FPS)
        writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), rate, size)

        for curr_frame_id in tqdm(range(self.total_frames)):
            self.mux.acquire()
            if self.writer_buf[curr_frame_id] is None:
                self.mux.release()
                curr_frame_id -= 1
                time.sleep(0.05)
                continue
            writer.write(self.writer_buf[curr_frame_id])
            self.buf_size -= self.writer_buf[curr_frame_id].nbytes
            self.writer_buf[curr_frame_id] = None
            self.mux.release()
        
        writer.release()


    def register_painter(self, func):
        """
        Register a painting function.

        Must have the prototype:
        ```python
        def func(frame: np.ndarray, frame_id: int) -> frame: np.ndarray
        ```
        """
        self.painter_funcs.append(func)


    def start(self, workers=3):
        """
        Render the video with multiple threads.
        """
        print(f"Rendering video to {self.output_path} with {workers} threads...")
        start_time = time.time()
        writer = threading.Thread(target=self.__writer, args=(self.output_path,))
        writer.start()

        for i in range(workers):
            t = threading.Thread(target=self.__renderer, args=(i,))
            t.setDaemon(True)
            t.start()

        writer.join() 

        self.cap.release()

        end_time = time.time()
        print(f"Rendering time: {end_time - start_time} seconds")
        
   