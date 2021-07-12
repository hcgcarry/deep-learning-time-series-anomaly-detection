
from numpy import array
# def split_sequence(sequence,windows_size,predict_w):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         # find the end of this pattern
#         end_ix = i + windows_size
#         out_end_ix = end_ix + predict_w
#         # check if we are beyond the sequence
#         if out_end_ix >= len(sequence):
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)

def split_sequence_multiStep(sequence,w,predict_w,n_features):
    X, y = list(), list()
    for i in range(0,len(sequence),predict_w):
        # find the end of this pattern
        end_ix = i + w 
        out_end_ix = end_ix + predict_w
        # check if we are beyond the sequence
        if out_end_ix >= len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    result_x = array(X)
    result_y = array(y)
    result_y = result_y.reshape((-1,n_features))
    return result_x,result_y