
# 引入库
from collections import (defaultdict, namedtuple)
from typing import (List, Tuple, Dict, Callable, Union)
import itertools
# from tqdm.notebook import tqdm
import tqdm
import warnings

from numba import jit
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.stats import ttest_1samp
from scipy.signal import (argrelmin, argrelmax)

import pandas as pd
import numpy as np

from multiprocessing import Pool

import matplotlib as mpl
import mplfinance as mpf
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置CPU工作核数
global CPU_WORKER_NUM
CPU_WORKER_NUM = 6




def rolling_windows(a: Union[np.ndarray, pd.Series, pd.DataFrame], window: int) -> np.ndarray:
    """
    Creates rolling-window 'blocks' of length `window` from `a`.
    Note that the orientation of rows/columns follows that of pandas.
    """

    if window > a.shape[0]:
        raise ValueError(
            "Specified `window` length of {0} exceeds length of"
            " `a`, {1}.".format(window, a.shape[0])
        )
    if isinstance(a, (pd.Series, pd.DataFrame)):
        a = a.values
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    windows = np.squeeze(
        np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    )
    if windows.ndim == 1:
        windows = np.atleast_2d(windows)
    return windows


def calc_smooth(prices: pd.Series, *, bw: Union[np.ndarray, str] = 'cv_ls', a: float = None, use_array: bool = True) -> Union[pd.Series, np.ndarray]:
    """计算Nadaraya–Watson核估计后的价格数据

    Args:
        prices (pd.Series): 价格数据
        bw (Union[np.ndarray,str]): Either a user-specified bandwidth or the method for bandwidth selection. Defaults to cv_ls.
        a (float, optional): 论文中所说的比例数据. Defaults to None.
        use_array (bool, optional): 为True返回ndarray,False返回为pd.Series. Defaults to True.

    Returns:
        Union[pd.Series,np.ndarry]
    """

    if not isinstance(prices, pd.Series):
        raise ValueError('prices必须为pd.Series')

    idx = np.arange(len(prices))

    kr = KernelReg(prices.values, idx,
                   var_type='c', reg_type='ll', bw=bw)

    if a is None:

        f = kr.fit(idx)[0]

    else:

        kr.bw = a * kr.bw 

        f = kr.fit(idx)[0]

    if use_array:

        return f

    else:

        return pd.Series(data=f, index=prices.index)

def find_price_argrelextrema(prices: pd.Series, *, bw: Union[str, np.ndarray] = 'cv_ls', a: float = 0.3, offset: int = 1, smooth_fumc: Callable = calc_smooth) -> pd.Series:
    """平滑数据并识别极大极小值

    Args:
        smooth_prices (pd.Series): 价格序列
        bw (Union[str,np.ndarray]): Either a user-specified bandwidth or the method for bandwidth selection. Defaults to cv_ls.
        a (float):论文中所说的比例数据. Defaults to 0.3.
        offset (int, optional): 避免陷入局部最大最小值. Defaults to 1.
        smooth_fumc (Callable,optional): 平滑处理方法函数. Defaults to calc_smooth
    Returns:
        pd.Series: 最大最小值的目标索引下标 index-dt value-price
    """
    size = len(prices)

    if size <= offset:
        raise ValueError('price数据长度过小')

    # 计算平滑价格
    smooth_arr: np.ndarray = smooth_fumc(prices, a=a, bw=bw, use_array=True)

    # 请多平滑后的高低点
    local_max = argrelmax(smooth_arr)[0]
    local_min = argrelmin(smooth_arr)[0]


    price_local_max_dt = []  # 储存索引下标

    for i in local_max:

        begin_idx = max(0, i-offset)
        end_idx = min(size, i+offset+1)
        price_local_max_dt.append(prices.iloc[begin_idx:end_idx].idxmax())

    price_local_min_dt = []  # 储存索引下标

    for i in local_min:

        begin_idx = max(0, i-offset)
        end_idx = min(size, i+offset+1)

        price_local_min_dt.append(prices.iloc[begin_idx:end_idx].idxmin())

    idx = (pd.to_datetime(price_local_max_dt + price_local_min_dt)
           .drop_duplicates()
           .sort_values())

    return prices.loc[idx]


def find_price_patterns(max_min: pd.Series, save_all: bool = True) -> Dict:
    """识别匹配常见形态,由于时间区间不同可能会有多个值

    Args:
        max_min (pd.Series): 识别后的数据
        save_all (bool, optional): 是否保留全部结果,True为保留全部,False保留首条数据. Defaults to True. 
        当窗口滑动时,历史上同一时间出现的形态可能会在多个连续窗口中被识别出来,为了不重复分析,我们只保留第一次识别到
        该形态的时点。
    Returns:
        Dict: 形态结果
    """

    if not isinstance(max_min, pd.Series):
        raise ValueError('max_min类型需要为pd.Series')

    patterns = defaultdict(list)  # 储存识别好的 形态信息
    size = len(max_min)


    if size < 5:
        return {}

    arrs: np.ndarray = rolling_windows(max_min, 5)  # 平滑并确定好高低点的价格数据
    idxs: np.ndarray = rolling_windows(np.array(max_min.index), 5)  # 索引

    for idx, arr in zip(idxs, arrs):

        # Head and Shoulders

        if _pattern_HS(arr):
            patterns['头肩顶(HS)'].append([(idx[0], idx[-1]), idx])

        # Inverse Head and Shoulders
        elif _pattern_IHS(arr):
            patterns['头肩底(IHS)'].append([(idx[0], idx[-1]), idx])

        # Broadening Top
        elif _pattern_BTOP(arr):
            patterns['顶部发散(BTOP)'].append([(idx[0], idx[-1]), idx])

        # Broadening Bottom
        elif _pattern_BBOT(arr):
            patterns['底部发散(BBOT)'].append([(idx[0], idx[-1]), idx])

        # Triangle Top
        elif _pattern_TTOP(arr):
            patterns['顶部收敛三角形(TTOP)'].append([(idx[0], idx[-1]), idx])

        # Triangle Bottom
        elif _pattern_TBOP(arr):
            patterns['底部收敛三角形(TBOT)'].append([(idx[0], idx[-1]), idx])

        # Rectangle Top
        elif _pattern_RTOP(arr):

            patterns['顶部矩形(RTOP)'].append([(idx[0], idx[-1]), idx])

        # Rectangle Bottom
        elif _pattern_RBOT(arr):
            patterns['底部矩形(RBOT)'].append([(idx[0], idx[-1]), idx])

        # TODO:双顶(DTOP),双底(DBOP)
        else:
            pass

    # 是否保留所有的形态识别
    if not save_all:
        # 仅保留区间内的
        tmp_dic = {}
        for k, v in patterns.items():
            tmp_dic[k] = v[0]
        patterns = tmp_dic

    return patterns




"""
形态定义
论文中e_1 is a maximum/minimum 但是HS和IHS应该是e_1与e_2比较大小(我是看论文图例得出的结论,否则找出的形态有问题)
"""


@jit(nopython=True)
def _pattern_HS(arr: np.ndarray) -> bool:
    """Head and Shoulders

    头肩顶:在上涨行情接近尾声时的看跌形态, 图形以左肩、头部、右肩及颈线构成。
    一般需 要经历连续的三次起落,也就是要出现三个局部极大值点。中间的高点比另
    外两个都高,称为头;左右两个相对较低的高点称为肩.
    """

    e1, e2, e3, e4, e5 = arr

    avg1 = np.array([e1, e5]).mean()
    avg2 = np.array([e2, e4]).mean()

    cond1 = (e1 > e2)  # (np.argmax(arr) == 0)
    cond2 = (e3 > e1) and (e3 > e5)
    cond3 = (0.985 * avg1 <= e1 <= avg1 *
             1.015) and (0.985 * avg1 <= e5 <= avg1 * 1.015)
    cond4 = (0.985 * avg2 <= e2 <= avg2 *
             1.015) and (0.985 * avg2 <= e4 <= avg2 * 1.015)

    return np.array([cond1, cond2, cond3, cond4]).all()


@jit(nopython=True)
def _pattern_IHS(arr: np.ndarray) -> bool:
    """Inverse Head and Shoulders

    头肩底:形态与头肩顶的方向刚好相反,需要经 历三个局部极小值点。进过头肩底形态之后,
    初始的下跌趋势会反转为上升趋势。
    """
    e1, e2, e3, e4, e5 = arr

    avg1 = np.array([e1, e5]).mean()
    avg2 = np.array([e2, e4]).mean()

    cond1 = (e1 < e2)  # (np.argmin(arr) == 0)
    cond2 = (e3 < e1) and (e3 < e5)
    cond3 = (0.985 * avg1 <= e1 <= avg1 *
             1.015) and (0.985 * avg1 <= e5 <= avg1 * 1.015)
    cond4 = (0.985 * avg2 <= e2 <= avg2 *
             1.015) and (0.985 * avg2 <= e4 <= avg2 * 1.015)

    return np.array([cond1, cond2, cond3, cond4]).all()


@jit(nopython=True)
def _pattern_BTOP(arr: np.ndarray) -> bool:
    """Broadening Top

    顶部发散:该形态由一极大值点开始,极大值和极小值点交替出现,而极大值点逐步抬高,极小值点逐
    步降低,波动逐渐增大。
    """
    e1, e2, e3, e4, e5 = arr

    cond1 = e1 > e2  # (np.argmax(arr) == 0)
    cond2 = (e1 < e3 < e5)
    cond3 = (e2 > e4)

    return np.array([cond1, cond2, cond3]).all()


@jit(nopython=True)
def _pattern_BBOT(arr: np.ndarray) -> bool:
    """Broadening Bottom

    底部发散:与顶部发散类似,同样都是极大值与极小值交替出现,高者愈高,低者愈低,区别在于底部
    发散形态的初始极值点为极小值点.
    """
    e1, e2, e3, e4, e5 = arr

    cond1 = e1 < e2  # (np.argmin(arr) == 0)
    cond2 = (e1 > e3 > e5)
    cond3 = (e2 < e4)

    return np.array([cond1, cond2, cond3]).all()


@jit(nopython=True)
def _pattern_TTOP(arr: np.ndarray) -> bool:
    """Triangle Top

    顶部收敛三角形:的初始为极大值点。在该形态下,价格的波动率逐渐减小.每轮波动的最高价都比前次低,
    而最低价都比前次高,呈现出收敛压缩图形.
    """
    e1, e2, e3, e4, e5 = arr

    cond1 = (np.argmax(arr) == 0)
    cond2 = (e1 > e3 > e5)
    cond3 = (e2 < e4)

    return np.array([cond1, cond2, cond3]).all()


@jit(nopython=True)
def _pattern_TBOP(arr: np.ndarray) -> bool:
    """Triangle Bottom

    底部收敛三角形:与顶部收敛三角形类似,区别在于初始为极小值点.
    """
    e1, e2, e3, e4, e5 = arr

    cond1 = (np.argmin(arr) == 0)
    cond2 = (e1 < e3 < e5)
    cond3 = (e2 > e4)

    return np.array([cond1, cond2, cond3]).all()


@jit(nopython=True)
def _pattern_RTOP(arr: np.ndarray) -> bool:
    """Rectangle Top

    矩形形态:为调整形态,即价格在某个区间内部上下波动,形态结束后会保持之前的趋势。
    顶部矩形由一极大值点开始,则波动之后会保持上涨趋势。
    """
    e1, e2, e3, e4, e5 = arr

    g1 = np.array([e1, e3, e5])
    g2 = np.array([e2, e4])

    rtop_g1 = np.mean(g1)
    rtop_g2 = np.mean(g2)

    cond1 = (np.argmax(arr) == 0)

    g1_ = np.abs(g1 - rtop_g1) / rtop_g1
    g2_ = np.abs(g2 - rtop_g2) / rtop_g2
    cond2 = np.all(g1_ <= 0.0075)
    cond3 = np.all(g2_ <= 0.0075)
    cond4 = np.min(g1) > np.max(g2)

    return np.array([cond1, cond2, cond3, cond4]).all()


@jit(nopython=True)
def _pattern_RBOT(arr: np.ndarray) -> bool:
    """Rectangle Bottom

    底部矩形由一极小值点开始,形态结束后保持下跌趋势
    """
    e1, e2, e3, e4, e5 = arr

    g1 = np.array([e1, e3, e5])
    g2 = np.array([e2, e4])

    rtop_g1 = np.mean(g1)
    rtop_g2 = np.mean(g2)

    cond1 = (np.argmin(arr) == 0)
    g1_ = np.abs(g1 - rtop_g1) / rtop_g1
    g2_ = np.abs(g2 - rtop_g2) / rtop_g2
    cond2 = np.all(g1_ < 0.0075)
    cond3 = np.all(g2_ < 0.0075)

    cond4 = np.min(g2) > np.max(g1)

    return np.array([cond1, cond2, cond3, cond4]).all()

# TODO:过于冗余 考虑使用多进程提升运行效率
# 初步完成多进程版本rolling_patterns2pool
def rolling_patterns(price: pd.Series, *, bw: Union[str, np.ndarray] = 'cv_ls', a: float = 0.3, n: int = 35, offset: int = 1, reset_window: int = None) -> namedtuple:
    """滑动窗口识别
       当窗口滑动时,历史上同一时间出现的形态可能会在多个连续窗口中被识别出来了不重复分析,我们只保留滑动期n内第一次识别到该形态的时点。
       当rest_window期后添加后续识别的形态
    Args:
        price (pd.Series): [description]
        bw (Union[str,np.ndarray]): Either a user-specified bandwidth or the method for bandwidth selection. Defaults to cv_ls.
        a (float):论文中所说的比例数据. Defaults to 0.3.
        n (int, optional): 计算窗口. Defaults to 35.
        offset (int, optional): 避免局部值的移动窗口. Defaults to 1.
        reset_window (int, optional): 自动的更新天数. Defaults to 窗口期.
    Returns:
        Dict: 匹配的结果
    """

    Record = namedtuple('Record', 'patterns,points')

    size = len(price)

    #
    if reset_window is None:

        reset_window = size - n + 1  # 表示不更新

    if reset_window >= size:

        raise ValueError('reset_window不能大于price长度')

    if offset <= 0 and offset >= size:

        raise ValueError('该参数必须为大于等于0且小于等于%s的数' % size)

    idxs: np.ndarray = rolling_windows(np.array(price.index), n)  # 按窗口期拆分时序

    patterns = defaultdict(list)  # 储存识别好的形态
    points = defaultdict(list)  # 出现形态时点

    for i, idx in enumerate(idxs):

        max_min: pd.Series = find_price_argrelextrema(price.loc[idx],
                                                      bw=bw,
                                                      a=a,
                                                      offset=offset)

        current_pattern: defaultdict = find_price_patterns(
            max_min, save_all=False)

        if current_pattern:

            if (i % reset_window == 0) and (i != 0):
                # 当大于更新长度时更新字典

                for k, v in current_pattern.items():

                    point, idx = v
                    patterns[k].append(point)  # 两点为识别出的形态区间
                    points[k].append(idx)  # 形态区间的五点位置

            else:

                # 当不是形态更新节点时 使用首次识别的形态
                keys = patterns.keys()
                # 当窗口滑动时,历史上同一时间出现的形态可能会在多个连续窗口中被识别出来，
                # 为了不重复分析，我们只保留第一次识别到该形态的时点。
                for k, v in current_pattern.items():

                    if k not in keys:
                        point, idx = v
                        patterns[k].append(point)  # 两点为识别出的形态区间
                        points[k].append(idx)  # 形态区间的五点位置
        else:

            continue

    record = Record(patterns=patterns, points=points)

    return record


def plot_patterns_chart(ohlc_data: pd.DataFrame, record_patterns: namedtuple, slice_range: bool = False, subplots: bool = False, ax=None):
    """标记识别出来的形态

    Args:
        ohlc_data (pd.DataFrame): 完整的行情数据OHLC
        record_patterns (namedtuple): 形态标记数据
        slice_range (bool, optional): True划分区间,False全行情展示. Defaults to False.
        subplots (bool, optional): 按形态名称子图划分. Defaults to False.

    Returns:
        [type]: [description]
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)

    COLORS = ['Crimson', 'DarkGoldenRod', 'DarkOliveGreen', 'DeepSkyBlue']
    if not record_patterns.patterns:
        raise ValueError('record_patterns为空')

    # 设置蜡烛图风格
    mc = mpf.make_marketcolors(up='r', down='g',
                               wick='i',
                               edge='i',
                               ohlc='i')

    s = mpf.make_mpf_style(marketcolors=mc)

    def _get_slice_price(tline: Union[Dict, np.array]) -> pd.DataFrame:
        """划分区间"""

        if isinstance(tline, dict):
            start_idx = ohlc_data.index.get_loc(tline['tlines'][0][0])
            end_idx = ohlc_data.index.get_loc(tline['tlines'][-1][-1])
            start = max(0, start_idx-25)
            end = min(len(ohlc_data), end_idx+30)

            return ohlc_data.iloc[start:end]

        else:
            start_idx = ohlc_data.index.get_loc(tline[0])
            end_idx = ohlc_data.index.get_loc(tline[-1])
            start = max(0, start_idx-25)
            end = min(len(ohlc_data), end_idx+30)
            # print(start,'   ',end)

            return ohlc_data.iloc[start:end]

    # 线段划分标记
    datepairs: List = []
    titles: List = []
    for title, dates in record_patterns.points.items():
        for d in dates:
            #dates = np.sort(np.array(list(record_patterns.point.values())).flatten())
            d = pd.to_datetime(d)
            datepair = [(d1, d2) for d1, d2 in zip(d, d[1:])]
            datepairs.append(datepair)
            titles.append(title)

    tlines = [dict(tlines=datepair, tline_use='close', colors=color, alpha=0.5, linewidths=5) for datepair,
              color in zip(datepairs, itertools.cycle(COLORS)) if datepair is not None]

    # 是否拆分画图
    if subplots:

        length = len(tlines)
        rows = int(np.ceil(length * 0.5))

        if ax is None:
            fig, axes = plt.subplots(rows, 2, figsize=(18, 3 * length))
        else:
            axes = ax

        axes = axes.flatten()

        for ax_i, (title, tline, ax) in enumerate(itertools.zip_longest(titles, tlines, axes)):

            if (ax_i == len(axes)-1) and (length % 2 != 0):

                ax.axis('off')
                break

            ax.set_title(title)

            if slice_range:

                mpf.plot(_get_slice_price(tline), style=s, tlines=tline,
                         type='candle', datetime_format='%Y-%m-%d', ax=ax)

            else:

                mpf.plot(ohlc_data, style=s, tlines=tline,
                         type='candle', datetime_format='%Y-%m-%d', ax=ax)

        plt.subplots_adjust(hspace=0.5)
        return axes

    else:

        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))

        if slice_range:

            all_dates: np.ndarray = np.array(
                [x for i in record_patterns.points.values() for x in i])
            all_dates = np.sort(np.unique(all_dates.flatten()))
            all_dates = pd.to_datetime(all_dates)

            mpf.plot(_get_slice_price(all_dates), style=s, tlines=tlines,
                     type='candle', datetime_format='%Y-%m-%d', ax=ax)
            return ax

        else:

            mpf.plot(ohlc_data, style=s, tlines=tlines,
                     type='candle', datetime_format='%Y-%m-%d', ax=ax)
        return ax

"""使用Multiprocessing"""

def _roll_patterns_series(arrs: List)->Tuple[int,defaultdict]:
    """获取窗口期内第一个匹配到的形态信息

    Args:
        arrs (List[np.ndarray,np.ndarray,int]): 0-价格,1-日期索引,3-切片的下标位置

    Returns:
        Tuple[int,defaultdict]: 0-切片下标信息 1-形态信息
    """
    slice_arr, idx_arr,id_num = arrs
    close_ser = pd.Series(data=slice_arr, index=idx_arr)

    max_min = find_price_argrelextrema(close_ser)

    return (id_num,find_price_patterns(max_min))


def rolling_patterns2pool(price: pd.Series,n:int,reset_window:int=None,n_workers: int = CPU_WORKER_NUM)->namedtuple:
    """使用多进程匹配

    Args:
        price (pd.Series): 价格数据
        n (int): 窗口期
        reset_window (int, optional): 字典更新窗口. Defaults to None.
        n_workers (int, optional): cpu工作数. Defaults to 4.

    Raises:
        ValueError: 基础检查

    Returns:
        namedtuple: _description_
    """
    size = len(price)
    if reset_window is None:

        reset_window = size - n + 1  # 表示不更新

    if reset_window >= size:

        raise ValueError('reset_window不能大于price长度')
        
    idxs: np.ndarray = rolling_windows(price.index.values, n)
    arr: np.ndarray = rolling_windows(price.values, n)
    id_num:np.ndarray = np.arange(len(idxs))
    chunk_size = calculate_best_chunk_size(len(idxs),n_workers)

    Record = namedtuple('Record', 'patterns,points')
    patterns = defaultdict(list)  # 储存识别好的形态
    points = defaultdict(list)  # 出现形态时点
    
    with Pool(processes=n_workers) as pool:

        res_tuple:Tuple[Tuple[int,Dict]] = tuple(pool.imap_unordered(_roll_patterns_series,
                    zip(arr, idxs,id_num), chunksize=chunk_size))

    res_tuple = sorted(res_tuple)

    for sub_res in res_tuple:

        num,current_pattern = sub_res

        if current_pattern:

            if (num % reset_window == 0) and (num != 0):
                # 当大于更新长度时更新字典

                for k, v in current_pattern.items():

                    point, idx = v[0]
                    patterns[k].append(point)  # 两点为识别出的形态区间
                    points[k].append(idx)  # 形态区间的五点位置

            else:

                # 当不是形态更新节点时 使用首次识别的形态
                keys = patterns.keys()
                # 当窗口滑动时,历史上同一时间出现的形态可能会在多个连续窗口中被识别出来，
                # 为了不重复分析，我们只保留第一次识别到该形态的时点。
                for k, v in current_pattern.items():

                    if k not in keys:
                        point, idx = v[0]
                        patterns[k].append(point)  # 两点为识别出的形态区间
                        points[k].append(idx)  # 形态区间的五点位置
        else:

            continue

    record = Record(patterns=patterns, points=points)

    return record


def calculate_best_chunk_size(data_length: int, n_workers: int) -> int:

    chunk_size, extra = divmod(data_length, n_workers * 5)
    if extra:
        chunk_size += 1
    return chunk_size

def To_time(date):

    import re
    dt_lst = []
    p = re.compile('[\u4e00-\u9fa5]')
    for dt in date:
        dt = p.split(dt)
        if len(dt[1])==1:
            dt[1] = '0'+dt[1]
        if len(dt[2])==1:
            dt[2] = '0'+dt[2]
        dt_lst.append(pd.to_datetime(dt[0]+'-'+dt[1]+'-'+dt[2]))
    return dt_lst

if __name__ == '__main__':

    baoan = pd.read_csv(r'000009.csv',encoding = 'gb18030').set_index('日期').iloc[::-1,:]
    baoan.index = To_time(baoan.index.values)
    baoan.columns = ['close','open','high','low','vol','rtn']
    baoan.tail(10)


    # res = rolling_patterns(baoan.close) 
    res = rolling_patterns2pool(baoan.close,10)
    plot_patterns_chart(baoan,res,True,False)


    print(res)
