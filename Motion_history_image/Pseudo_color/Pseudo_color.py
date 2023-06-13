import cv2
import numpy as np
import os
import sys
import glob
sys.path.append('../')
from Motion_history_image.Motion_History_Image import MotionHistoryImage
from Motion_history_image.Motion_History_Image_Pseudo_color import MotionHistoryImage_Pseudo

colormap_table_count = 0
colormap_table = [
    ['COLORMAP_AUTUMN',  cv2.COLORMAP_AUTUMN ],
    ['COLORMAP_JET',     cv2.COLORMAP_JET    ],
    ['COLORMAP_WINTER',  cv2.COLORMAP_WINTER ],
    ['COLORMAP_RAINBOW', cv2.COLORMAP_RAINBOW],
    ['COLORMAP_OCEAN',   cv2.COLORMAP_OCEAN  ],
    ['COLORMAP_SUMMER',  cv2.COLORMAP_SUMMER ],
    ['COLORMAP_SPRING',  cv2.COLORMAP_SPRING ],
    ['COLORMAP_COOL',    cv2.COLORMAP_COOL   ],
    ['COLORMAP_HSV',     cv2.COLORMAP_HSV    ],
    ['COLORMAP_PINK',    cv2.COLORMAP_PINK   ],
    ['COLORMAP_HOT',     cv2.COLORMAP_HOT    ]
]

video_input = cv2.VideoCapture(0)
MHI= MotionHistoryImage_Pseudo()

MHI(SaveFlag=False)
