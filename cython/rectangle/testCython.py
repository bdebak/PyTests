# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 19:26:34 2018

@author: bdebaque
"""

# %%
import rect
x0, y0, x1, y1 = 1, 2, 3, 4
rect_obj = rect.PyRectangle(x0, y0, x1, y1)
print((rect_obj.get_area()))