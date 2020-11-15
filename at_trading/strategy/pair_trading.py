import threading
import time
from ibapi.contract import Contract
import pandas as pd
import matplotlib.pyplot as plt

from at_trading.ib.ib_api import ATIBApi, TickType

if __name__ == '__main__':
    app = ATIBApi()
    app.connect_local(port=7497)
    # app.connect_local()
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    app.wait_till_connected()
    goog = Contract()
    goog.symbol = 'GOOG'
    goog.secType = 'STK'
    goog.exchange = 'SMART'
    goog.currency = 'USD'

    appl = Contract()
    appl.symbol = 'AAPL'
    appl.secType = 'STK'
    appl.exchange = 'SMART'
    appl.currency = 'USD'
    # non blocking
    # app.req_mkt_data(contract, snapshot=False)

    app.req_tick_by_tick_data(goog, TickType.mid)
    app.req_tick_by_tick_data(appl, TickType.mid)

    time.sleep(30)
    for n in range(2):
        time.sleep(30)
        resp_ts = {}
        for k, v in iter(app.resp_dict.items()):
            sec_symbol = app.req_dict[k]['func_param']['contract'].symbol
            sec_df = pd.DataFrame(v)
            sec_df['in_time'] = pd.to_datetime(sec_df['in_time'], unit='s')
            sec_ts = sec_df[['in_time', 'mid_point']].set_index('in_time')['mid_point']
            sec_ts = sec_ts.resample('3S').mean()
            resp_ts[sec_symbol] = sec_ts
        resp_df = pd.DataFrame(resp_ts)
        resp_df = resp_df.pct_change()
        diff_ts = resp_df['GOOG'] - resp_df['AAPL']
        diff_ts - diff_ts.std()
    app.disconnect()
