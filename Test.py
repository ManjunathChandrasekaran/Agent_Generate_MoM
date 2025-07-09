import ctypes
try:
    ctypes.WinDLL("cublas64_12.dll")
    print("cuBLAS loaded successfully")
except OSError as e:
    print("cuBLAS load failed:", e)