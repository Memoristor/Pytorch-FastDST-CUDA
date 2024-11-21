#coding=utf-8


from scipy import fftpack as fp
import torch
import time
import cv2

import fadst


def load_image(fpath: str):
    """Load image as a contiguous torch.Tensor
    """
    image = cv2.imread(fpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).float().contiguous()
    image = image.cuda()
    return image
    

def pretty_tag(tag: str):
    """Format tags using pretty comments 
    """
    res = '\n'
    res += '#' * 100 + '\n'
    res += '# ' + '\n'
    res += f'# {tag}' + '\n'
    res += '# ' + '\n'
    res += '#' * 100 + '\n'
    return res


def check_error(error: torch.Tensor):
    """Check the min error, max error, mean error.
    """
    res = True
    if error.min() > 1e-3:
        print(f'[Warning] The minimum error, {error.min():4.4f}, is too large')
        res = False
    else:
        print(f'[Info] The minimum error is {error.min():4.4f}')
    if error.max() > 1e-3:
        print(f'[Warning] The maximum error, {error.max():4.4f}, is too large')
        res = False
    else:
        print(f'[Info] The maximum error is {error.max():4.4f}')
    if error.mean() > 1e-3:
        print(f'[Warning] The average error, {error.mean():4.4f}, is too large')
        res = False
    else:
        print(f'[Info] The average error is {error.mean():4.4f}')
    return res


def dct2d(input: torch.Tensor, point: int):
    """2D DCT using scipy
    """
    for i in range(point):
        input[i, :] = fp.dct(input[i, :], norm='ortho')
    for i in range(point):
        input[:, i] = fp.dct(input[:, i], norm='ortho')
    return input


def patch_dct2d(input: torch.Tensor, point: int):
    """2D block-wise DCT using scipy
    """
    [C, H, W] = input.shape
    for c in range(C):
        for h in range(int(H / point)):
            for w in range(int(W / point)):
                patch = input[c, h * point: (h + 1) * point, w * point: (w + 1) * point]
                input[c, h * point: (h + 1) * point, w * point: (w + 1) * point] = dct2d(patch, point)
    return input


if __name__ == '__main__':
    
    x = load_image('assets/image.png')
    block = 8

    print(pretty_tag('test x -> DCT(x), error = abs(DCT_CUDA(x) - DCT_CPU(x))'))
        
    start_cuda = time.time()
    dct2d_cuda = fadst.naiveDCT2D(x, block)
    end_cuda = time.time()

    start_cpu = time.time()
    dct2d_cpu = patch_dct2d(x.cpu().numpy(), block)
    dct2d_cpu = torch.from_numpy(dct2d_cpu).float().cuda()
    error = torch.abs(dct2d_cuda - dct2d_cpu)
    end_cpu = time.time()

    print(f'[Info] CUDA cost {end_cuda - start_cuda :4.6f} s')
    print(f'[Info] CPU cost {end_cpu - start_cpu :4.6f} s')
    check_error(error)
    
    
    print(pretty_tag('test DCT(x) -> x, error = abs(x - IDCT(DCT(x)))'))
        
    naive_dct_2d = fadst.naiveDCT2D(x, block)
    naive_idct_2d = fadst.naiveIDCT2D(naive_dct_2d, block)
    error = torch.abs(naive_idct_2d - x)

    iterations = 100
    total_time = 0
    for i in range(iterations):
        start = time.time()
        output = fadst.naiveDCT2D(x, block)
        end = time.time()
        total_time += end - start
        
    print(f'[Info] CUDA average cost {total_time / iterations * 1e6:4.6f} us')
    check_error(error)
    
    
    print(pretty_tag('test DST(x) -> x, error = abs(x - IDST(DST(x)))'))
        
    naive_dst_2d = fadst.naiveDST2D(x, block)
    naive_idst_2d = fadst.naiveIDST2D(naive_dst_2d, block)
    error = torch.abs(naive_idst_2d - x)

    iterations = 100
    total_time = 0
    for i in range(iterations):
        start = time.time()
        output = fadst.naiveDST2D(x, block)
        end = time.time()
        total_time += end - start
        
    print(f'[Info] CUDA average cost {total_time / iterations * 1e6:4.6f} us')
    check_error(error)
    

    print(pretty_tag('test DHT(x) -> x, error = abs(x - IDHT(DHT(x)))'))
        
    naive_dht_2d = fadst.naiveDHT2D(x, block)
    naive_idht_2d = fadst.naiveIDHT2D(naive_dht_2d, block)
    error = torch.abs(naive_idht_2d - x)

    iterations = 100
    total_time = 0
    for i in range(iterations):
        start = time.time()
        output = fadst.naiveDHT2D(x, block)
        end = time.time()
        total_time += end - start
        
    print(f'[Info] CUDA average cost {total_time / iterations * 1e6:4.6f} us')
    check_error(error)


    print(pretty_tag('test DCT(x) -> SORT(DCT(x)), error = abs(DCT(x) - RECOVER(SORT(DCT(x))))'))
            
    sort_dct = fadst.sortCoefficients(naive_dct_2d, block)
    recover_dct = fadst.recoverCoefficients(sort_dct, block)
    error = torch.abs(naive_dct_2d - recover_dct)
    
    check_error(error)
    

    print(pretty_tag('test y = RECOVER(SORT(DCT(x))), y -> IDCT(y), error = abs(x - IDCT(y))'))
            
    recover_idct = fadst.naiveIDCT2D(recover_dct, block)
    error = torch.abs(x - recover_idct)
    
    check_error(error)
    

    print('\n')
    print('Test completed!')
    print('\n')

