# Description: An example of how to use the Renderer class to render a video with custom painting functions.
import cv2 as cv

from utils.renderer import Renderer

if __name__ == "__main__":
    # 1. Create a renderer object with the video path
    renderer = Renderer("example.mp4", "output.mp4")

    # 2. Define the painting functions
    def example_painter_1(frame, frame_id):
        # Draw a text on the frame
        cv.putText(frame, f"Frame ID: {frame_id}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        return frame
    
    def example_painter_2(frame, frame_id):
        # Draw a rectangle on the frame
        cv.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), 2)
        return frame
    
    # 3. Register the painting functions
    # The order of registration determines the order of execution
    renderer.register_painter(example_painter_1) 
    renderer.register_painter(example_painter_2)

    # 4. Start the rendering process
    renderer.start()