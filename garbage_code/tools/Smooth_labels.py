# # 标签平滑
# def smooth_labels(y, smooth_factor=0.1):
#     assert len(y.shape) == 2
#     if 0 <= smooth_factor <= 1:
#         y *= 1 - smooth_factor
#         y += smooth_factor / y.shape[1]
#     else:
#         raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))
#     return y
