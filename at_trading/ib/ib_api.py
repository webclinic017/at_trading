import time
import datetime
from collections import OrderedDict, defaultdict
import signal  # doesn't work on windows
from typing import List, Dict
from functools import reduce

from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.client import EClient, SmartComponentMap, TickAttribLast, TickAttribBidAsk, \
    BarData, HistogramData
from ibapi.contract import Contract, ContractDetails
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.utils import current_fn_name
from ibapi.wrapper import EWrapper, TickTypeEnum
import threading
import numpy as np
import logging
import enum

# *
# constants
# *
REAL_TIME_BAR_TYPES = ['TRADES',
                       'BID',
                       'ASK',
                       'MIDPOINT'
                       ]


class TickType(enum.Enum):
    bid_ask = 'BidAsk'
    last = 'Last'
    all_last = 'AllLast'
    mid = 'MidPoint'


class MarketDataType(enum.Enum):
    live = 1
    frozen = 2
    delayed = 3
    delayed_frozen = 4


# *
# contracts/i.e. Assets/Securities
# *

def gen_contract(symbol, sec_type, exchange, currency):
    """
    shortcut for creating a contract

    :param symbol:
    :param sec_type:
    :param exchange:
    :param currency:
    :return:
    """
    result_contract = Contract()
    result_contract.symbol = symbol
    result_contract.secType = sec_type
    result_contract.exchange = exchange
    result_contract.currency = currency
    return result_contract


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def blocking(func, timeout_seconds=3):
    def _blocking_decorator(self, *args, **kwargs):
        logger = logging.getLogger(__name__)
        data_result = None
        try:
            with timeout(timeout_seconds):
                req_id = func(self, *args, **kwargs)
                if req_id is None:
                    raise Exception(f'req_id is not returned')
                if req_id not in self.req_dict:
                    raise Exception(f'req_id [{req_id}] does not exist in request dictionary')
                while self.req_dict[req_id]['req_status'] != 'DONE':
                    time.sleep(0.1)
                if req_id not in self.resp_dict:
                    raise Exception(f'req_id [{req_id}] does not exist in response dictionary')
                data_result = self.resp_dict[req_id]
        except Exception as e:
            logger.error(e)
        return data_result

    return _blocking_decorator


def camel_to_sneak(input_str):
    return reduce(lambda x, y: x + ('_' if y.isupper() else '') + y, input_str).lower()


# *
# main classes to overwrite as an application
# *
# TODO: overwrite these behaviour

class ATIBApi(EWrapper, EClient):
    def __init__(self, log_blocking_req: bool = False, env_type='live'):
        """
        usage:
        >>> app = ATIBApi()
        >>> app.connect_local()

        >>> api_thread = threading.Thread(target=app.run, daemon=True)
        >>> api_thread.start()


        """
        EClient.__init__(self, self)
        self.env_type = env_type
        # there are two types of request, blocking vs non-blocking
        # if this flag is turned off, then only log non-blocking requests
        self.log_blocking_req = log_blocking_req
        # TODO: keep all the requests in memory for now, though we should use logs/db
        # request_dict is in the format of
        # {123 (request id): {'req_type':'subscribe_asset', 'req_timestamp':}}
        self.req_dict = OrderedDict()
        self.resp_dict = defaultdict(list)
        self.next_order_id = None
        self.order_dict = OrderedDict()
        self.account_dict = dict()
        self.account_history = OrderedDict()

        self.account_summary = dict()
        self.account_summary_history = []

        self.position_list = []
        self.position_history = []

    # helper functions
    def wait_till_connected(self, timeout_seconds=2):
        logger = logging.getLogger(__name__)
        try:
            with timeout(timeout_seconds):
                while not self.isConnected() or self.next_order_id is None:
                    time.sleep(1)
                logger.info('connected to server')
        except Exception as e:
            logger.error(e)

    def nextValidId(self, order_id: int):
        print(f'next valid id gets called [{order_id}]')
        super().nextValidId(order_id)
        self.next_order_id = order_id

    def connect_local(self, port=7496, client_id=0):
        self.connect('127.0.0.1', port, client_id)

    def gen_req_id(self):
        result_id = 0
        if len(self.req_dict) > 0:
            result_id = np.max(list(self.req_dict.keys())) + 1
        return result_id

    def log_req(self, req_id, func_name, func_params):
        params = dict(func_params)
        for delete_item in ['req_id', 'self']:
            if delete_item in params:
                del params[delete_item]
        self.req_dict[req_id] = {'func_name': func_name,
                                 'func_param': params,
                                 'req_timestamp': datetime.datetime.now(),
                                 'req_status': 'STARTED',
                                 'req_done_timestamp': None
                                 }

    def update_resp_list(self, req_id, input_data: List, update_status=True):
        if update_status:
            self.req_dict[req_id]['req_status'] = 'DONE'

        if req_id not in self.resp_dict:
            self.resp_dict[req_id] = input_data
        else:
            self.resp_dict[req_id].extend(input_data)

    def update_resp_obj(self, req_id, input_data: object, update_status=True):
        if update_status:
            self.req_dict[req_id]['req_status'] = 'DONE'

        self.resp_dict[req_id].append(input_data)

    def cancel_all_req(self):
        self.cancel_mkt_depth_all()
        self.cancel_tick_by_tick_all()
        self.cancel_mkt_data_all()
        self.req_dict = OrderedDict()

    def clean_all(self):
        self.cancel_all_req()
        self.resp_dict = defaultdict(list)

    # *
    # non-blocking requests!
    # *

    # reqMktData: request functions
    def req_mkt_data(self, contract: Contract, generic_tick_list='',
                     snapshot=False, regulatory_snapshot=False, mkt_data_options=None):
        """

        :param contract:
        :param generic_tick_list: consists of list of tick fields requested
        https://interactivebrokers.github.io/tws-api/tick_types.html

        :param snapshot:
        :param regulatory_snapshot:
        :param mkt_data_options:
        """
        req_id = self.gen_req_id()
        if mkt_data_options is None:
            _mk_options = []
        else:
            _mk_options = mkt_data_options
        self.reqMktData(req_id, contract, generic_tick_list, snapshot, regulatory_snapshot, _mk_options)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    # reqMktData: response functions
    def tickPrice(self, req_id, tick_type, price, attrib):
        print(f'{req_id}|{TickTypeEnum.idx2name[tick_type]}|{price}|{attrib}')

    def tickSize(self, req_id: int, tick_type: int, size: int):
        print(f'{req_id}|{TickTypeEnum.idx2name[tick_type]}|{size}')

    def tickString(self, req_id, tick_type: int, value: str):
        print(f'{req_id}|{TickTypeEnum.idx2name[tick_type]}|{value}')

    def tickSnapshotEnd(self, req_id: int):
        print(f'{req_id} snapshot end')

    def tickGeneric(self, req_id: int, tick_type: int, value: float):
        print(f'{req_id}|{TickTypeEnum.idx2name[tick_type]}|{value}')

    def tickReqParams(self, ticker_id: int, min_tick: float, bbo_exchange: str, snapshot_permissions: int):
        print(f'{ticker_id}|{min_tick}|{bbo_exchange}|{snapshot_permissions}')

    def cancel_mkt_data(self, req_id: int):
        print(f'request {req_id} is cancelled')
        self.cancelMktData(req_id)
        self.req_dict[req_id]['req_status'] = 'CANCELLED'

    def cancel_mkt_data_all(self):
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_mkt_data':
                self.cancel_mkt_data(k)

    # reqTickByTickData: request functions
    def req_tick_by_tick_data(self, contract: Contract, tick_type: TickType, ignore_size: bool = True):
        req_id = self.gen_req_id()
        self.reqTickByTickData(req_id, contract, tick_type.value, 0, ignore_size)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def tickByTickAllLast(self, req_id: int, tick_type: int, in_time: int, price: float,
                          size: int, tick_attrib_last: TickAttribLast, exchange: str,
                          special_conditions: str):
        print(f'{req_id}|{tick_type}|{in_time}|{price}|{size}|{tick_attrib_last}|{exchange}|{special_conditions}')

    def tickByTickBidAsk(self, req_id: int, in_time: int, bid_price: float, ask_price: float,
                         bid_size: int, ask_size: int, tick_attrib_bid_ask: TickAttribBidAsk):
        print(f'{req_id}|{in_time}|{bid_price}|{ask_price}|{bid_size}|{ask_size}|{tick_attrib_bid_ask}')

    def tickByTickMidPoint(self, req_id: int, in_time: int, mid_point: float):
        print(f'{req_id}|{in_time}|{mid_point}')

    def cancel_tick_by_tick_data(self, req_id: int):
        print(f'request {req_id} is cancelled')
        self.cancelTickByTickData(req_id)
        self.req_dict[req_id]['req_status'] = 'CANCELLED'

    def cancel_tick_by_tick_all(self):
        print(f'cancel all tick by tick request')
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_tick_by_tick_data' and v['req_status'] == 'STARTED':
                self.cancel_tick_by_tick_data(k)

    # switch between market data type
    def req_market_data_type(self, market_data_type: int):
        req_id = self.gen_req_id()
        self.reqMarketDataType(market_data_type)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def marketDataType(self, req_id: int, market_data_type: int):
        print(f'{req_id}| switched to market data type {MarketDataType(market_data_type).name}')

    def req_mkt_depth(self, contract, num_rows: int, is_smart_depth: bool, mkt_depth_options: List):
        req_id = self.gen_req_id()
        self.reqMktDepth(req_id, contract, num_rows, is_smart_depth, mkt_depth_options)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    # REFERENCE: Market depth will be returned via the
    # IBApi.EWrapper.updateMktDepth or the IBApi.EWrapper.updateMktDepthL2 callback.
    # The two functions differ in that when there is a market maker or exchange identifier
    # to be returned market depth data will be relayed back through IBApi.EWrapper.updateMktDepthL2.
    # Otherwise it is returned to IBApi.EWrapper.updateMktDepth.
    # For example, ARCA only has ARCA itself as a Market Maker.
    # Therefore when requesting market depth data from ARCA, the data will be relayed back
    # via IBApi.EWrapper.updateMktDepth. On the other hand, with ISLAND (the ECN for NASDAQ)
    # market maker information is provided, so the market maker MPID will be relayed back via
    # IBApi.EWrapper.updateMktDepthL2. The market maker MPID is reported in the 'marketMaker'
    # string argument of the callback function.

    def updateMktDepth(self, req_id: int, position: int, operation: int,
                       side: int, price: float, size: int):
        print(f'{req_id}|{position}|{operation}|{side}|{price}|{size}')

    def updateMktDepthL2(self, req_id: int, position: int, market_maker: str,
                         operation: int, side: int, price: float, size: int, is_smart_depth: bool):
        print(f'{req_id}|{position}|{market_maker}|{operation}|{side}|{price}|{size}|{is_smart_depth}')

    def cancel_mkt_depth(self, req_id: int, is_smart_depth: bool = True):
        print(f'request {req_id} is cancelled')
        self.cancelMktDepth(req_id, is_smart_depth)
        self.req_dict[req_id]['req_status'] = 'CANCELLED'

    def cancel_mkt_depth_all(self):
        print(f'cancel all tick by tick request')
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_mkt_depth' and v['req_status'] == 'STARTED':
                self.cancel_mkt_depth(k)

    def req_real_time_bars(self, contract: Contract, bar_size: int,
                           what_to_show: str, use_rth: bool,
                           real_time_bars_options: List):
        req_id = self.gen_req_id()
        assert what_to_show in REAL_TIME_BAR_TYPES, f'what_to_show {what_to_show} is not supported'
        self.reqRealTimeBars(req_id, contract, bar_size, what_to_show, use_rth, real_time_bars_options)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def realtimeBar(self, req_id: int, in_time: int, open_px: float, high_px: float, low_px: float, close_px: float,
                    volume: int, wap: float, count: int):
        print(f'{req_id}|{in_time}|{open_px}|{high_px}|{low_px}|{close_px}|{volume}|{wap}|{count}')

    def cancel_real_time_bars(self, req_id):
        print(f'request {req_id} is cancelled')
        self.cancelRealTimeBars(req_id)
        self.req_dict[req_id]['req_status'] = 'CANCELLED'

    def cancel_real_time_bars_all(self):
        print(f'cancel all tick by tick request')
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_real_time_bars' and v['req_status'] == 'STARTED':
                self.cancel_mkt_depth(k)

    # two types of historical bar data, if keep_up_to_date = True and do not pass in an end date,
    # then it becomes non-blocking
    # otherwise blocking
    def req_historical_data_non_blocking(self, contract: Contract,
                                         duration_str: str, bar_size_setting: str, what_to_show: str,
                                         use_rth: int, format_date: int, chart_options: List):
        req_id = self.gen_req_id()
        self.reqHistoricalData(req_id, contract, '', duration_str, bar_size_setting, what_to_show,
                               use_rth, format_date, True, chart_options)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    @blocking
    def req_historical_data_blocking(self, contract: Contract, end_date_time: str,
                                     duration_str: str, bar_size_setting: str, what_to_show: str,
                                     use_rth: int, format_date: int, chart_options: List):
        req_id = self.gen_req_id()
        self.reqHistoricalData(req_id, contract, end_date_time, duration_str, bar_size_setting, what_to_show,
                               use_rth, format_date, False, chart_options)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def historicalData(self, req_id: int, bar: BarData):
        if req_id not in self.resp_dict:
            self.resp_dict[req_id] = [bar]
        else:
            self.resp_dict[req_id].append(bar)
        print(f'{req_id}|{bar}')

    def historicalDataUpdate(self, req_id: int, bar: BarData):
        print(f'{req_id}|{bar}')

    def historicalDataEnd(self, req_id: int, start: str, end: str):
        self.req_dict[req_id]['req_status'] = 'DONE'
        print(f'{req_id} historical data [{start}-{end}] done')

    def cancel_historical_data(self, req_id: int):
        print(f'request {req_id} is cancelled')
        self.cancelHistoricalData(req_id)
        self.req_dict[req_id]['req_status'] = 'CANCELLED'

    def cancel_historical_data_all(self):
        print(f'cancel all historical request')
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_historical_data_blocking' and v['req_status'] == 'STARTED':
                self.cancel_mkt_depth(k)

    def req_histogram_data(self, contract, use_rth, time_period):
        req_id = self.gen_req_id()
        self.reqHistogramData(req_id, contract, use_rth, time_period)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def histogramData(self, req_id: int, items: HistogramData):
        print(f'{req_id}|{HistogramData}')

    def cancel_histogram_data(self, req_id: int):
        print(f'request {req_id} is cancelled')
        self.cancelHistogramData(req_id)
        self.req_dict[req_id]['req_status'] = 'CANCELLED'

    def cancel_histogram_data_all(self):
        print(f'cancel all histogram request')
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_histogram_data' and v['req_status'] == 'STARTED':
                self.cancel_mkt_depth(k)

    # orders
    def place_order(self, contract: Contract, input_order: Order):
        # we don't know if order id can be the same as request id
        req_id = self.gen_req_id()
        order_id = self.next_order_id
        self.placeOrder(order_id, contract, input_order)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def openOrder(self, order_id: int, contract: Contract, input_order: Order,
                  order_state: OrderState):
        super().openOrder(order_id, contract, input_order, order_state)
        # print(f'{order_id}|{contract}|{input_order}|{order_state.__dict__}')
        assert input_order.permId != 0, 'cannot handle order with permid = 0'
        # use perm id as key instead and don't include everything
        include_items = [
            'clientId',
            'orderId',
            'account',
            'action',
            'orderType',
            'totalQuantity',
            'cashQty',
            'lmtPrice',
            'auxPrice'
        ]

        order_to_log = {camel_to_sneak(k): v for k, v in iter(input_order.__dict__.items())
                        if k in include_items}
        order_to_log['status'] = order_state.status
        self.order_dict[input_order.permId] = order_to_log

    def orderStatus(self, order_id: int, status: str, filled: float,
                    remaining: float, avg_fill_price: float, perm_id: int,
                    parent_id: int, last_fill_price: float, client_id: int,
                    why_held: str, mkt_cap_price: float):
        super().orderStatus(order_id, status, filled, remaining,
                            avg_fill_price, perm_id,
                            parent_id, last_fill_price, client_id, why_held, mkt_cap_price)

        print(f'{order_id}|{status}|{filled}|{remaining}|{avg_fill_price}|{perm_id}|' +
              f'{parent_id}|{last_fill_price}|{client_id}|{why_held}|{mkt_cap_price}')

        for param_k, param_v in iter(vars().items()):
            if param_k not in ['__class__', 'self', 'perm_id']:
                self.order_dict[perm_id][param_k] = param_v

    def clean_orders(self):
        self.order_dict = OrderedDict()

    def cancel_order(self, order_id):
        self.cancelOrder(order_id)

    def req_global_cancel(self):
        self.reqGlobalCancel()

    def req_account_updates(self, subscribe: bool, acct_code: str):
        req_id = self.gen_req_id()
        self.reqAccountUpdates(subscribe, acct_code)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def cancel_account_updates(self, acct_code: str):
        self.req_account_updates(False, acct_code)
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_account_updates' and v['req_status'] in ['STARTED', 'DONE']:
                self.req_dict[k]['req_status'] = 'CANCELLED'

    def clean_account(self):
        self.account_dict = dict()
        self.account_history = OrderedDict()

    def updateAccountValue(self, key: str, val: str, currency: str,
                           account_name: str):
        val_to_store = {k: v for k, v in iter(vars().items()) if k not in ['self']}
        self.account_dict['account_value'] = val_to_store

    def updateAccountTime(self, time_stamp: str):
        self.account_dict['account_time'] = time_stamp

    def updatePortfolio(self,
                        contract: Contract,
                        position: float,
                        market_price: float,
                        market_value: float,
                        average_cost: float,
                        unrealized_pnl: float,
                        realized_pnl: float,
                        account_name: str):
        val_to_store = {k: v for k, v in iter(vars().items()) if k not in ['self']}
        if 'positions' not in self.account_dict:
            self.account_dict['positions'] = [val_to_store]
        else:
            self.account_dict['positions'].append(val_to_store)

    def accountDownloadEnd(self, account_name: str):
        super().accountDownloadEnd(account_name)
        print(f'account download end [{account_name}]')
        self.account_history[self.account_dict['account_time']] = self.account_dict
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_account_updates':
                self.update_resp_obj(k, self.account_dict, True)
        self.account_dict = dict()

    def req_account_summary(self, group_name: str, tags: str):
        req_id = self.gen_req_id()
        self.reqAccountSummary(req_id, group_name, tags)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def accountSummary(self, req_id: int, account: str, tag: str, value: str,
                       currency: str):
        val_to_store = {k: v for k, v in iter(vars().items()) if k not in ['self']}
        if req_id not in self.account_summary:
            self.account_summary[req_id] = [val_to_store]
        else:
            self.account_summary[req_id].append(val_to_store)

    def accountSummaryEnd(self, req_id: int):
        acct_sum = self.account_summary[req_id]
        self.account_summary_history.append(acct_sum)
        self.update_resp_obj(req_id, acct_sum, True)
        del self.account_summary[req_id]

    def cancel_account_summary(self, req_id: int):
        self.cancelAccountSummary(req_id)
        self.req_dict[req_id]['req_status'] = 'CANCELLED'

    def cancel_account_summary_all(self):
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_account_summary' and v['req_status'] in ['STARTED', 'DONE']:
                self.cancel_account_summary(k)

    def req_positions(self):
        req_id = self.gen_req_id()
        self.reqPositions()
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def position(self, account: str, contract: Contract, position: float,
                 avg_cost: float):
        val_to_store = {k: v for k, v in iter(vars().items()) if k not in ['self']}
        self.position_list.append(val_to_store)

    def positionEnd(self):
        self.position_history.append(self.position_list)
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_positions' and v['req_status'] == 'STARTED':
                self.update_resp_obj(k, self.position_list, True)

        self.position_list = []

    def cancel_positions(self):
        self.cancelPositions()
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_positions' and v['req_status'] in ['STARTED', 'DONE']:
                self.req_dict[k]['req_status'] = 'CANCELLED'

    def req_pnl_single(self, input_account: str, model_code: str, contract_id: int):
        req_id = self.gen_req_id()
        self.reqPnLSingle(req_id, input_account, model_code, contract_id)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def pnlSingle(self, req_id: int, pos: int, daily_pnl: float, unrealized_pnl: float,
                  realized_pnl: float, value: float):
        val_to_store = {k: v for k, v in iter(vars().items()) if k not in ['self', 'req_id']}
        val_to_store['pnl_timestamp'] = datetime.datetime.now()
        self.update_resp_obj(req_id, val_to_store, True)

    def cancel_pnl_single(self, req_id):
        self.cancelPnLSingle(req_id)
        self.req_dict[req_id]['req_status'] = 'CANCELLED'

    def req_pnl(self, account: str, model_code: str):
        req_id = self.gen_req_id()
        self.reqPnL(req_id, account, model_code)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def pnl(self, req_id: int, daily_pnl: float, unrealized_pnl: float, realized_pnl: float):
        val_to_store = {k: v for k, v in iter(vars().items()) if k not in ['self', 'req_id']}
        val_to_store['pnl_timestamp'] = datetime.datetime.now()
        self.update_resp_obj(req_id, val_to_store, True)

    def cancel_pnl(self, req_id):
        self.cancelPnL(req_id)
        self.req_dict[req_id]['req_status'] = 'CANCELLED'

    # *
    # blocking requests!
    # *
    # we convert some real time request to blocking for convenience
    # account update
    @blocking
    def req_pnl_blocking(self, account: str, model_code: str):
        req_id = self.req_pnl(account, model_code)
        return req_id

    @blocking
    def req_pnl_single_blocking(self, input_account: str, model_code: str, contract_id: int):
        req_id = self.req_pnl_single(input_account, model_code, contract_id)
        return req_id

    @blocking
    def req_account_updates_blocking(self, acct_code):
        req_id = self.req_account_updates(True, acct_code)
        return req_id

    # account summary
    @blocking
    def req_account_summary_blocking(self, group_name: str, tags: str):
        req_id = self.req_account_summary(group_name, tags)
        return req_id

    # positions
    @blocking
    def req_positions_blocking(self):
        req_id = self.req_positions()
        return req_id

    @blocking
    def req_open_orders(self):
        # this is a refresh of the open orders
        self.clean_orders()
        req_id = self.gen_req_id()
        self.reqOpenOrders()
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def openOrderEnd(self):
        super().openOrderEnd()
        # change the status of the request to be done
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_open_orders' and v['req_status'] == 'STARTED':
                self.update_resp_obj(k, self.order_dict, True)

    @blocking
    def req_managed_accts(self):
        req_id = self.gen_req_id()
        self.reqManagedAccts()
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def managedAccounts(self, accounts_list: str):
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_managed_accts' and v['req_status'] == 'STARTED':
                self.update_resp_obj(k, accounts_list, True)

    @blocking
    def req_contract_details(self, contract: Contract):
        req_id = self.gen_req_id()
        self.reqContractDetails(req_id, contract)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def contractDetails(self, req_id: int, contract_details: ContractDetails):
        self.update_resp_obj(req_id, contract_details, False)

    def contractDetailsEnd(self, req_id: int):
        self.req_dict[req_id]['req_status'] = 'DONE'
        print(f'{req_id} contract detail done')

    @blocking
    def req_matching_symbols(self, pattern):
        req_id = self.gen_req_id()
        self.reqMatchingSymbols(req_id, pattern)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def symbolSamples(self, req_id: int,
                      contract_description: List):
        print(f'{req_id}|{contract_description}')
        # for functions that don't have 'End' callback, log here
        self.update_resp_list(req_id, contract_description, True)

    @blocking
    def req_smart_components(self, exchange_char):
        """
            The tick types 'bidExch' (tick type 32), 'askExch' (tick type 33), 'lastExch' (tick type 84)
            are used to identify the source of a quote. To find the full exchange name corresponding to a
            single letter code returned in tick types 32, 33, or 84, and API function
            IBApi::EClient::reqSmartComponents is available.
        """
        req_id = self.gen_req_id()
        self.reqSmartComponents(req_id, exchange_char)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def smartComponents(self, req_id: int, smart_component_map: SmartComponentMap):
        self.update_resp_obj(req_id, smart_component_map, True)

    @blocking
    def req_mkt_depth_exchanges(self):
        req_id = self.gen_req_id()
        self.reqMktDepthExchanges()
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def mktDepthExchanges(self, depth_mkt_data_descriptions: List):
        # no req_id, we just need to find it from req_dict
        req_id = None
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_mkt_depth_exchanges':
                req_id = k
                break

        if req_id is not None:
            if req_id not in self.resp_dict:
                self.resp_dict[req_id] = depth_mkt_data_descriptions
            else:
                self.resp_dict[req_id].extend(depth_mkt_data_descriptions)

    @blocking
    def req_historical_ticks(self, contract: Contract, start_date_time: str,
                             end_date_time: str, number_of_ticks: int, what_to_show: str, use_rth: int,
                             ignore_size: bool, misc_options: List):
        req_id = self.gen_req_id()
        self.reqHistoricalTicks(req_id, contract, start_date_time, end_date_time,
                                number_of_ticks, what_to_show, use_rth, ignore_size, misc_options)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def historicalTicks(self, req_id: int, ticks: List, done: bool):
        self.update_resp_list(req_id, ticks, done)

    def historicalTicksBidAsk(self, req_id: int, ticks: List, done: bool):
        self.update_resp_list(req_id, ticks, done)

    def historicalTicksLast(self, req_id: int, ticks: List, done: bool):
        self.update_resp_list(req_id, ticks, done)

    @blocking
    def req_head_time_stamp(self, contract: Contract,
                            what_to_show: str, use_rth: int, format_date: int):
        req_id = self.gen_req_id()
        self.reqHeadTimeStamp(req_id, contract,
                              what_to_show, use_rth, format_date)
        self.log_req(req_id, current_fn_name(), vars())
        return req_id

    def headTimestamp(self, req_id: int, head_time_stamp: str):
        self.update_resp_obj(req_id, head_time_stamp, True)

    def cancel_head_time_stamp(self, req_id: int):
        print(f'request {req_id} is cancelled')
        self.cancelHeadTimeStamp(req_id)
        self.req_dict[req_id]['req_status'] = 'CANCELLED'

    def cancel_head_time_stamp_all(self):
        print(f'cancel all head time stamp requests')
        for k, v in iter(self.req_dict.items()):
            if v['func_name'] == 'req_head_time_stamp' and v['req_status'] == 'STARTED':
                self.cancel_head_time_stamp(k)


if __name__ == '__main__':
    app = ATIBApi()
    # app.connect_local(port=7497)
    app.connect_local()
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    app.wait_till_connected()
    asset = Contract()
    asset.symbol = 'IBM'
    asset.secType = 'STK'
    asset.exchange = 'SMART'
    asset.currency = 'USD'

    # non blocking
    # app.req_mkt_data(asset, snapshot=False)

    # app.req_tick_by_tick_data(asset, TickType.mid)
    # app.req_mkt_depth(asset, 10, True, [])
    # app.req_real_time_bars(asset, 1, 'TRADES', True, [])
    # app.req_account_updates(True, 'DU1589832')
    # result = app.req_historical_data_non_blocking(asset, "1 D", '1 min', 'MIDPOINT', 1, 1, [])
    # app.req_account_summary('All', AccountSummaryTags.AllTags)

    # blocking
    # result = app.req_contract_details(asset)
    # result = app.req_matching_symbols('IBM')
    # result = app.req_mkt_depth_exchanges()
    # queryTime = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime("%Y%m%d %H:%M:%S")
    # result = app.req_historical_data_blocking(asset, queryTime, "1 M", '1 day', 'MIDPOINT', 1, 1, [])
    # result = app.req_historical_ticks(asset, "20200622 10:39:33", '', 100, 'TRADES', 1, True, [])
    # result2 = app.req_head_time_stamp(asset, 'TRADES', 0, 1)

    # orders: be very careful of this
    # order = Order()
    # order.action = 'BUY'
    # order.orderType = 'LMT'
    # order.totalQuantity = 100
    # order.lmtPrice = 80
    #
    # # app.clean_orders()
    # result = app.req_open_orders()
    # # app.place_order(asset, order)
    #
    # # app.req_global_cancel()
    # result2 = app.req_managed_accts()
    # account = app.req_account_updates_blocking('DU1589832')
    # account_summary = app.req_account_summary_blocking('All', AccountSummaryTags.AllTags)
    # positions = app.req_positions_blocking()

    # app.req_pnl_single('U1069514', '', 388013150)
    # pnl_result = app.req_pnl_single_blocking('U1069514', '', 388013150)
    app.req_pnl('U1069514', '')
    time.sleep(5)
    # app.cancel_mkt_data_all()
    app.disconnect()
