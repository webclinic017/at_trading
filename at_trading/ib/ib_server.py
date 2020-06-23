import threading

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper
import pandas as pd
from flask import Flask, jsonify, request
import logging
import pandas as pd
from flask_cors import CORS
import time
from at_trading.ib.ib_api import ATIBApi

app = Flask(__name__)
CORS(app)
app_ib = ATIBApi()
app_ib.connect_local()
api_thread = threading.Thread(target=app_ib.run, daemon=True)
api_thread.start()
time.sleep(1)


@app.route('/subscribe/<ticker>', methods=['GET'])
def subscribe_to_asset(ticker):
    logger = logging.getLogger(__name__)
    try:
        contract = Contract()
        contract.symbol = ticker
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        app_ib.req_mkt_data(contract)
    except Exception as e:
        logger.error(e)
    return jsonify('success')


# driver function
if __name__ == '__main__':
    # Start the socket in a thread
    api_thread = threading.Thread(target=app_ib.run, daemon=True)
    api_thread.start()
    app.run(debug=False)
