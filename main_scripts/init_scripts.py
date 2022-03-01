import sys
import os


def init_script():
    # makes the tvg package visible no matter where the scripts
    # are launched from    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(script_dir))
