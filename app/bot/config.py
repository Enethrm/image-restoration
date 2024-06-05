import os
from configparser import ConfigParser


cfg = ConfigParser() 
cfg.read(os.path.join(os.path.dirname(__file__), 'config.ini'),encoding='utf-8')

