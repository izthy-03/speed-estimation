import pysrt
import re


class SRT_reader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.srt = pysrt.open(filepath)
        self.length = len(self.srt)

    
    def get_diff_time_ms(self, frame_id):
        """
        Get the duration of the frame in milliseconds
        
        Args:
            frame_id (int): The frame id
            
        Returns:
            int: The duration of the frame in milliseconds
        """
        if frame_id < 0 or frame_id >= self.length:
            return None
        
        return self.srt.data[frame_id].duration.milliseconds

    
    def search_text(self, frame_id, regex):
        """
        Search the text in the frame matches the regex
        
        Args:
            frame_id (int): The frame id
            regex (str): The regex to match
            
        Returns:
            string: The first matched text in the frame
            bool: True if the text matches the regex, False otherwise
        """
        if frame_id < 0 or frame_id >= self.length:
            return None, False

        res = re.search(regex, self.srt.data[frame_id].text) 
        if res is None:
            return None, False

        return res.group(1), True

    
    def get_int(self, frame_id, key):
        """
        Get the integer value of the key in the frame. 
        The key is in the format of "key: value"
        """
        text, match = self.search_text(frame_id, rf"{key}: ([-]?[0-9.]+)")
        if not match:
            return None
        return int(text)

    def get_float(self, frame_id, key):
        """
        Get the float value of the key in the frame. 
        The key is in the format of "key: value"
        """
        text, match = self.search_text(frame_id, rf"{key}: ([-]?[0-9.]+)")
        if not match:
            return None
        return float(text)
        

    