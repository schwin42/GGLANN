import win32ui
import ctypes

#Find window titles
EnumWindows = ctypes.windll.user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
GetWindowText = ctypes.windll.user32.GetWindowTextW
GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
IsWindowVisible = ctypes.windll.user32.IsWindowVisible
 
titles = []
def foreach_window(hwnd, lParam):
    if IsWindowVisible(hwnd):
        length = GetWindowTextLength(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        GetWindowText(hwnd, buff, length + 1)
        titles.append(buff.value)
    return True
EnumWindows(EnumWindowsProc(foreach_window), 0)
 
print(titles)


# window_name = "Target Window Name" # use EnumerateWindow for a complete list
# wd = win32ui.FindWindow(None, window_name)
# dc = wd.GetWindowDC() # Get window handle
# j = dc.GetPixel (60,20)  # as practical and intuitive as using PIL!
# print j
# dc.DeleteDC() # necessary to handle garbage collection, otherwise code starts to slow down over many iterations