# -*- coding: utf-8 -*-

import logging
import MySQLdb
import numpy as np
import os
import pandas.io.sql as pandas_sql
import pandas as pd
import re
import warnings

from collections import defaultdict
from functools import partial
from sqlalchemy import event, create_engine
from sqlalchemy import exc
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker



####################################################
# sql functions
####################################################


def db_execute(db_session, sql_str):
    _result = db_session.execute(sql_str)
    db_session.commit()
    db_session.remove()


def add_engine_pidguard(engine):
    """
    Add multiprocessing guards.

    Forces a connection to be reconnected if it is detected
    as having been shared to a sub-process.
    """

    @event.listens_for(engine, "connect")
    def connect(dbapi_connection, connection_record):
        connection_record.info['pid'] = os.getpid()

    @event.listens_for(engine, "checkout")
    def checkout(dbapi_connection, connection_record, connection_proxy):
        pid = os.getpid()
        if connection_record.info['pid'] != pid:
            # substitute log.debug() or similar here as desired
            warnings.warn(
                "Parent process %(orig)s forked (%(newproc)s) with an open "
                "database connection, "
                "which is being discarded and recreated." %
                {"newproc": pid, "orig": connection_record.info['pid']})
            connection_record.connection = connection_proxy.connection = None
            raise exc.DisconnectionError(
                "Connection record belongs to pid %s, "
                "attempting to check out in pid %s" %
                (connection_record.info['pid'], pid)
            )


def gen_connection(host, user, password, db_name, db_type='mysql'):
    """
    generate a connection object for different database types
    :param host: host of the db server
    :param user: username
    :param password: password
    :param db_name: database name
    :param db_type: only support mysql now
    :return: return connection object
    """

    _conn = None
    if db_type == 'mysql':
        _conn = MySQLdb.connect(host, user, password, db_name)
        # _conn.set_character_set('utf8')
    return _conn


def gen_engine(in_host, in_user, in_pass, in_db_name, in_db_engine_type, in_pool_recycle=3600, in_pool_size=10):
    conn_bh = partial(gen_connection,
                      host=in_host,
                      user=in_user,
                      password=in_pass,
                      db_name=in_db_name,
                      db_type=in_db_engine_type)

    db_engine = create_engine("{0}://".format(in_db_engine_type), creator=conn_bh, pool_recycle=in_pool_recycle,
                              pool_size=in_pool_size, encoding="utf-8")
    return db_engine


def gen_session(in_host, in_user, in_pass, in_db_name, in_db_engine_type, in_pool_recycle=3600, in_pool_size=10):
    db_engine = gen_engine(in_host, in_user, in_pass, in_db_name, in_db_engine_type, in_pool_recycle=in_pool_recycle,
                           in_pool_size=in_pool_size)
    add_engine_pidguard(db_engine)
    _session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=db_engine))
    return _session


def _fix_query_str(sql_str):
    """
    # fix an issue withe query having % within
    # for instance query = '%%usd%%'
    :param sql_str:
    :return:
    usage:
        >>> sql_str = 'blah%%dfk%% somethingelse% again%something %blahblah'
        >>> _fix_query_str(sql_str)
    """

    if '%' in sql_str:
        assert '|' not in sql_str, 'both % and | in query cannot handle'
        _temp_query = re.sub('%+', '|', sql_str)
        sql_str = _temp_query.replace('|', '%%')
    return sql_str


def db_select(db_session, sql_str, return_df=True, pd_native=True):
    """
    :param db_session:
    :param sql_str:
    :param return_df:
    :param pd_native:
    :return:

    usage:
        >>> sql_str = 'select * from mapping_country'
        >>> from bhshared.db import db_session_bh
        >>> db_select(db_session_bh, sql_str, return_df=True, pd_native=True)
    """

    if pd_native:
        if '%' in sql_str:
            sql_str = _fix_query_str(sql_str)
        _result = pandas_sql.read_sql(sql_str, con=db_session.bind)
    else:
        _result = db_session.execute(sql_str)
        if return_df:
            _result = pd.DataFrame(_result.fetchall(), columns=[x[0] for x in _result._cursor_description()])
        else:
            _result = _result.fetchall()
    db_session.remove()
    return _result


def db_insert_df(db_session, table_name, input_df, schema=None, skip_error=False):
    """
    Insert DataFrame into database

    :param db_session:
    :param table_name:
    :param input_df:
    :param schema:
    :param skip_error:
    :return: 0 if no errors
    -1 if skip_error=True and error is encountered
    """

    _logger = logging.getLogger(__name__)
    _status = 0

    if skip_error is True:
        try:
            input_df.to_sql(table_name, db_session.bind, if_exists='append',
                            index=False) if schema is None else \
                input_df.to_sql(table_name, db_session.bind, if_exists='append', index=False,
                                schema=schema)
            _logger.info(log_df_operation(log_prefix='inserted to', table_name=table_name,
                                          df=input_df, log_df_content=False))
        except exc.IntegrityError:  # use pymysql.err.IntegrityError for pymysql if needed
            _logger.info(log_df_operation(log_prefix='encountered SQL integrity error, skipping insertion to',
                                          table_name=table_name, df=input_df, log_df_content=False))
            _status = -1
    else:
        input_df.to_sql(table_name, db_session.bind, if_exists='append',
                        index=False) if schema is None else \
            input_df.to_sql(table_name, db_session.bind, if_exists='append', index=False,
                            schema=schema)
        _logger.info(log_df_operation(log_prefix='inserted to', table_name=table_name,
                                      df=input_df, log_df_content=False))

    db_session.commit()
    db_session.remove()
    return _status


def db_create_table(db_session, tbl_name, input_df,
                    pk_list=None, fk_dict=None, add_time_stamp=True, schema=None):
    """

    :param add_time_stamp:
    :param pk_list:
    :param fk_dict:
        this should be a dictionary where key is the existing table's primary key and it should reference to
        another table and specify the key mappings

    :param tbl_name:
    :param schema:
    :param db_session:
    :param input_df:

    - string
    - unicode
    - bytes
    - floating
    - integer
    - mixed-integer
    - mixed-integer-float
    - decimal
    - complex
    - categorical
    - boolean
    - datetime64
    - datetime
    - date
    - timedelta64
    - timedelta
    - time
    - period
    - mixed

    CREATE TABLE `mapping_asset` (
      # attributes
      `bh_asset_id` varchar(100) COLLATE utf8_bin NOT NULL,
      `bh_type` varchar(100) COLLATE utf8_bin DEFAULT NULL,
      `name` varchar(255) COLLATE utf8_bin DEFAULT NULL,
      `full_name` varchar(255) COLLATE utf8_bin DEFAULT NULL,
      `country_iso` varchar(10) COLLATE utf8_bin DEFAULT NULL,
      `time_stamp` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

      # primary key
      PRIMARY KEY (`bh_asset_id`)

      # contraints
      CONSTRAINT `fk_fund_asset_id` FOREIGN KEY (`bh_asset_id`)
      REFERENCES `mapping_asset` (`bh_asset_id`) ON DELETE CASCADE ON UPDATE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;


    usage:
        >>> bbg_ticker = 'BIV US Equity'
        >>> bh_type = 'fund'
        >>> input_df = get_attr_bbg(bbg_ticker, bh_type)['asset_fund']
        >>> db_session = db_session_bh
        >>> pk_list = 'bh_asset_id'
        >>> fk_dict = {'bh_asset_id':('mapping_asset', 'bh_asset_id')}
        >>> add_time_stamp=True
        >>> schema = 'bh-rover'
        >>> tbl_name = 'bin_test'
    """

    # """      `bh_asset_id` varchar(100) COLLATE utf8_bin NOT NULL,
    #   `bh_type` varchar(100) COLLATE utf8_bin DEFAULT NULL,
    #   `name` varchar(255) COLLATE utf8_bin DEFAULT NULL,
    #   `full_name` varchar(255) COLLATE utf8_bin DEFAULT NULL,
    #   `country_iso` varchar(10) COLLATE utf8_bin DEFAULT NULL,
    #   """
    _dtype_ts = pd.Series([pd.api.types.infer_dtype(input_df[x], skipna=True) for x in input_df.columns],
                          index=input_df.columns)
    _len_ts = input_df.fillna('').astype(str).apply(lambda x: x.str.len()).mean()
    _bins = [0, 50, 100, 10000000]
    _varchar_len = ['VARCHAR(100)', 'VARCHAR(255)', 'TEXT']
    if pk_list is not None:
        _pk_list = make_list(pk_list)
    else:
        _pk_list = []

    _attr_list = []

    for _col in input_df.columns:
        if _dtype_ts[_col] in ['string', 'unicode', 'bytes', 'mixed']:
            _len = _len_ts[_col]
            _idx = np.digitize(_len, _bins).tolist()
            _dtype = _varchar_len[_idx - 1]
        elif _dtype_ts[_col] in ['floating', 'decimal', 'mixed-integer-float']:
            _dtype = 'DOUBLE'
        elif _dtype_ts[_col] in ['integer', 'mixed-integer']:
            _dtype = 'BIGINT(20)'
        elif _dtype_ts[_col] in ['date']:
            _dtype = 'DATE'
        elif _dtype_ts[_col] in ['datetime64', 'datetime']:
            _dtype = 'DATETIME'
        elif _dtype_ts[_col] in ['time']:
            _dtype = 'TIME'
        elif _dtype_ts[_col] in ['boolean']:
            _dtype = 'TINYINT(4)'
        else:
            raise Exception('type [{}] not supported'.format(_dtype_ts[_col]))

        _nullable = 'DEFAULT NULL'
        if pk_list is not None and _col in _pk_list:
            _nullable = 'NOT NULL'
        _attr = "`{}` {} COLLATE utf8_bin {}".format(_col, _dtype, _nullable)
        _attr_list.append(_attr)
    # add timestamp
    if add_time_stamp:
        _ts_attr = '`time_stamp` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'
        _attr_list.append(_ts_attr)

    # add pk
    if pk_list is not None:
        _pk = ','.join(['`{}`'.format(x) for x in _pk_list])
        _pk_attr = 'PRIMARY KEY ({})'.format(_pk)
        _attr_list.append(_pk_attr)

    # add fk
    if fk_dict is not None:
        assert isinstance(fk_dict, dict), 'fk_dict has to be a dictionary'

        for _k, _v in iter(fk_dict.items()):
            assert len(_v) == 2, 'fk target should be tuple with (tablename, fieldname)'
            _fk_attr = """CONSTRAINT `fk_{}_{}` FOREIGN KEY (`{}`)
            REFERENCES `{}` (`{}`) ON DELETE CASCADE ON UPDATE CASCADE""".format(tbl_name, _k, _k, _v[0], _v[1])
            _attr_list.append(_fk_attr)

    _body_create = ','.join(_attr_list)
    if schema is None:
        _tabl = '`{}`'.format(tbl_name)
    else:
        _tabl = '`{}`.`{}`'.format(schema, tbl_name)

    _sql_base = """CREATE TABLE IF NOT EXISTS {} ({}) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;""".format(
        _tabl,
        _body_create)
    db_execute(db_session, _sql_base)


SQL_VALID_EXIST_HANDLE = ['ignore', 'delete_insert',
                          'update_insert_using_partial_snapshot',
                          'update_insert_using_full_snapshot']


def convert_date_to_str_df(df):
    # convert date to str for input df
    if df.empty:
        return df

    for _x in df.columns:
        _val_x = df[_x]
        if is_date(_val_x[0]):
            _val_x_date_str = [convert_date_to_str(pd.to_datetime(x)) for x in _val_x]
            df.loc[:, _x] = _val_x_date_str
    return df


def db_insert_df_handle_exist(db_session, table_name, input_df, pk_col_list,
                              exist_handle='ignore', schema=None, skip_error=False, db_update=True):
    """
    :param db_session:
    :param table_name:
    :param input_df:
    :param pk_col_list:
    :param exist_handle:
    :param schema:
    :param skip_error:
    :param db_update: If True, perform actual database operations
    usage:
        >>> from bhshared.db.db_session import db_session_bh
        >>> db_session = db_session_bh
        >>> table_name = 'test_ts_fund_px'
        >>> input_df = [
        ['1924-07-15', 'fund|mitdx|us5757368304', 1.1, 6],
        ['1924-07-15', 'fund|mittx|us5757361036', 2, None],
        ['1924-07-31', 'test1', None, 8],
        ['1924-07-31', 'fund|mittx|us5757361036', 4, 9.9],
        ['1924-08-31', 'test2', 5, 10],
        ]
        >>> cls = ['date', 'bh_asset_id', 'mid', 'nav']
        >>> input_df = pd.DataFrame.from_records(input_df, columns=cls)
        >>> pk_col_list = ['date', 'bh_asset_id']
        >>> exist_handle = 'update_insert_using_full_snapshot'
        >>> schema = 'bh-rover'
        >>> skip_error = False
        >>> db_update = False

        >>> db_insert_df_handle_exist(db_session, table_name, input_df, pk_col_list,
        exist_handle, schema, skip_error, db_update)
    """

    _logger = logging.getLogger(__name__)
    assert exist_handle in SQL_VALID_EXIST_HANDLE, 'exist_handle [{}] not supported'.format(exist_handle)
    _co_list = make_list(pk_col_list)
    # find if data already exist in the database
    _col_sql = ','.join(_co_list)

    if schema is not None:
        _sql = 'select * from `{}`.{}'.format(schema, table_name)
    else:
        _sql = 'select * from {}'.format(table_name)
    _sql = sql_append_criteria(in_sql=_sql, select_criteria=input_df[_co_list])

    result_dict = defaultdict(pd.DataFrame)
    _df_ful = db_select(db_session, _sql)
    _df = _df_ful.loc[:, _co_list]
    _df_ful = _df_ful.loc[:, [x for x in _df_ful.columns if x != 'time_stamp']]
    result_dict['df_ful'] = _df_ful
    _df_ful = convert_date_to_str_df(_df_ful)
    input_df.reset_index(drop=True, inplace=True)
    input_df = convert_date_to_str_df(input_df)
    if not _df.empty:
        # filter out dfs matching particular criteria
        _q_and = sql_append_criteria(in_sql='', select_criteria=_df)
        _q_and = _q_and.replace(' where', '')
        _df_exist = input_df.query(_q_and)
        # df1.isin(df2) returns False when comparing None to None, do not use on non-primary keys!
        _df_not_exist = input_df.loc[~input_df[_co_list].isin(_df_exist[_co_list]).all(axis=1)]
    else:
        _df_exist = pd.DataFrame()
        _df_not_exist = input_df
    result_dict['df_exist'] = _df_exist
    result_dict['df_not_exist'] = _df_not_exist

    # sort for later
    _df_ful.sort_values(_co_list, inplace=True)
    _df_ful.reset_index(drop=True, inplace=True)

    _to_insert = None
    if exist_handle == 'ignore':
        _logger.info(log_df_operation('ignored the following IDs since they already exist in',
                                      table_name, _df_exist, log_df_content=True))
        _to_insert = _df_not_exist
        result_dict['to_insert'] = _to_insert

    elif exist_handle == 'delete_insert':
        if not _df_exist.empty:
            if schema is None:
                _sql = 'delete from {}'.format(table_name)
            else:
                _sql = 'delete from `{}`.{}'.format(schema, table_name)
            _sql = sql_append_criteria(in_sql=_sql, select_criteria=_df_exist[_co_list])
            if db_update:
                db_execute(db_session, _sql)
                _logger.info(log_df_operation('deleted from', table_name, _df_exist, log_df_content=False))
        _to_insert = input_df
        result_dict['to_insert'] = _to_insert

    elif exist_handle == 'update_insert_using_partial_snapshot':
        _to_insert = _df_not_exist
        result_dict['to_insert'] = _to_insert

        if not _df_exist.empty:
            _df_replaced = _df_ful.merge(_df_exist, on=_co_list, how='left', suffixes=['', '_new'])

            # sort for later
            _df_replaced.sort_values(_co_list, inplace=True)
            _df_replaced.reset_index(drop=True, inplace=True)

            # the below block is the key diff between full snapshot update vs. partial snapshot update
            _df_ix = pd.DataFrame(columns=[x for x in _df_exist.columns if x not in _co_list])
            for _x in _df_ix.columns:
                _x_new = f'{_x}_new'
                _ix = (_df_replaced[_x] != _df_replaced[_x_new]) & (~pd.isnull(_df_replaced[_x_new]))
                _df_replaced.loc[_ix, _x] = _df_replaced.loc[_ix, _x_new]
                _df_ix[_x] = _ix
            _df_replaced = _df_replaced[_df_ful.columns]

            # here is where previous two sorts matter
            _to_replace = _df_replaced.loc[_df_ix.any(axis=1)]  # new content that will replace existing db content
            _to_be_replaced = _df_ful.loc[_df_ix.any(axis=1)]  # existing db content that will be replaced
            result_dict['to_replace'] = _to_replace
            result_dict['to_be_replaced'] = _to_be_replaced

    elif exist_handle == 'update_insert_using_full_snapshot':
        _to_insert = _df_not_exist
        result_dict['to_insert'] = _to_insert

        if not _df_exist.empty:
            _df_replaced = _df_ful.merge(_df_exist, on=_co_list, how='left', suffixes=['', '_new'])

            # sort for later
            _df_replaced.sort_values(_co_list, inplace=True)
            _df_replaced.reset_index(drop=True, inplace=True)

            # the below block is the key diff between full snapshot update vs. partial snapshot update
            _df_ix = pd.DataFrame(columns=[x for x in _df_ful.columns if x not in _co_list])
            for _x in _df_ix.columns:
                _x_new = f'{_x}_new'
                # new content does not contain existing attribute _x in db, need to
                if _x_new not in _df_replaced.columns:
                    _df_replaced[_x_new] = None
                _ix = (_df_replaced[_x] != _df_replaced[_x_new])
                _ix &= ~(pd.isnull(_df_replaced[_x]) & pd.isnull(_df_replaced[_x_new]))
                _df_replaced.loc[_ix, _x] = _df_replaced.loc[_ix, _x_new]
                _df_ix[_x] = _ix
            _df_replaced = _df_replaced[_df_ful.columns]

            # here is where previous two sorts matter
            _to_replace = _df_replaced.loc[_df_ix.any(axis=1)]  # new content that will replace existing db content
            _to_be_replaced = _df_ful.loc[_df_ix.any(axis=1)]  # existing db content that will be replaced
            result_dict['to_replace'] = _to_replace
            result_dict['to_be_replaced'] = _to_be_replaced

    # replace existing db content that's different from new content
    if 'to_replace' in result_dict.keys():
        if _to_replace is not None and not _to_replace.empty:
            for i in _to_replace.index:
                if schema is None:
                    _sql_table = table_name[:]
                else:
                    _sql_table = f'`{schema}`.{table_name}'
                _sql_select = f'select * from {_sql_table}'
                _sql_replace = f'update {_sql_table}'
                _sql_where = sql_append_criteria(in_sql='', select_criteria=_to_replace.loc[[i], _co_list])

                # make sure only one row matches the select criteria
                _sql_select += _sql_where
                _db_existing_content = db_select(db_session, _sql_select)
                assert len(_db_existing_content) == 1, \
                    log_df_operation('more than one existing records identified, please check select_criteria',
                                     table_name, _db_existing_content, log_df_content=True)

                # append update content
                _sql_replace = sql_append_update(in_sql=_sql_replace,
                                                 update_content=_to_replace.loc[[i],
                                                                                [x for x in _to_replace.columns
                                                                                 if x not in _co_list]])
                # append select criteria
                _sql_replace += _sql_where

                # update existing db content
                if db_update:
                    db_execute(db_session, _sql_replace)

            if db_update:
                _logger.debug(log_df_operation('updated', table_name, _to_replace, log_df_content=True))

    # insert new entries into db
    if _to_insert is not None and not _to_insert.empty:
        if db_update:
            db_insert_df(db_session, table_name, _to_insert, schema=schema, skip_error=skip_error)

    return result_dict


def log_df_operation(log_prefix, table_name, df, log_df_content=True):
    if df.empty:
        return ''
    else:
        if log_df_content:
            log = '{} table: {}; columns: {}; content: '.format(log_prefix, table_name, df.columns.tolist())
            for i in df.index:
                log += '{}, '.format(df.loc[i].values.tolist())
            log = log[:-len(', ')]
        else:
            log = '{} table: {}; number of rows impacted: {}'.format(log_prefix, table_name, len(df))
        return log


def sql_append_criteria(in_sql, select_criteria=None):
    """
    :param in_sql:
    :param select_criteria:
    :return:
    """

    assert type(in_sql) == str, f'in_sql [{in_sql}] has to be of str type'

    if select_criteria is not None:
        assert isinstance(select_criteria, dict) | isinstance(select_criteria, pd.DataFrame), \
            f'non-empty select_criteria [{select_criteria}] has to be of dict or df type'

    if isinstance(select_criteria, dict):  # dict can have non-matching lengths
        if select_criteria:  # returns True if dictionary is not empty
            out_sql = in_sql + ' where '
            for column, criterion in iter(select_criteria.items()):
                if isinstance(criterion, str):
                    out_sql += '{} = \'{}\' and '.format(column, criterion)
                elif criterion is not None:
                    out_sql += '{} in ({}) and '.format(column, ','.join(
                        sorted(['\'{}\''.format(y) for y in set(criterion) if y is not None])))
            out_sql = out_sql[:-len(' and ')]

    elif isinstance(select_criteria, pd.DataFrame):
        if not select_criteria.empty:
            # convert date to str
            for x in select_criteria.columns:
                if is_date(select_criteria[x].values.tolist()[0]):
                    select_criteria.loc[:, x] = [convert_date_to_str(pd.to_datetime(y)) for y in select_criteria[x]]

            detail_criteria = ' and '.join(
                ["{} in ({})".format(x,
                                     ','.join(sorted(['\'{}\''.format(y)
                                                      for y in select_criteria[x].unique() if y is not None])))
                 for x in select_criteria.columns])
            out_sql = in_sql + f' where {detail_criteria}'

    # address issue of empty in, such as:
    # select * from `bh-rover`.ts_fund_global_rank where
    # bh_asset_id in ('fund|35024178|gb0005292593') and end_date in ('2019-08-31','2019-09-13') and period in
    # ('1D','1Y') and metric in ('DistributionYield','Lipper Distribution Yield','PriceChange',
    # 'ProjectedYield','SimpleYieldBegin','SimpleYieldEnd','Yield') and currency in ()
    out_sql = out_sql.replace('in ()', 'in ("")')

    return out_sql


def sql_append_update(in_sql, update_content=None):
    """
    :param in_sql:
    :param update_content:
    :return:
    """

    assert type(in_sql) == str, f'in_sql [{in_sql}] has to be of str type'

    if update_content is not None:
        assert isinstance(update_content, dict) | isinstance(update_content, pd.DataFrame), \
            f'non-empty update_content [{update_content}] has to be of dict or df type'

    # convert None to null
    none2null = lambda x: 'null' if x is None else x

    # quote manager
    quotemgr = lambda x: x.replace('"', "'") if isinstance(x, str) else x

    if isinstance(update_content, dict):  # dict can have non-matching lengths
        if update_content:  # returns True if dictionary is not empty
            out_sql = in_sql + ' set '
            for column, content in iter(update_content.items()):
                if isinstance(content, str):
                    out_sql += '{} = \'{}\', '.format(column, content)
                else:
                    out_sql += '{} = \'{}\', '.format(column, sorted(list(set(none2null(content))))[0])
            out_sql = out_sql[:-len(', ')]

    elif isinstance(update_content, pd.DataFrame):  # only 1st entry in each column gets used
        if not update_content.empty:
            detail_criteria = ', '.join(
                ['{} = "{}"'.format(x, quotemgr(none2null(update_content[x].values.tolist()[0])))
                 for x in update_content.columns])
            out_sql = in_sql + f' set {detail_criteria}'

    # remove quotes around null
    out_sql = out_sql.replace('= "null"', '= null')

    # convert nan to null
    out_sql = out_sql.replace('= "nan"', '= null').replace("= 'nan'", '= null')

    return out_sql


def db_select_by_criteria(db_session, table_name, select_column, select_criteria=None):
    """
    :param db_session:
    :param table_name:
    :param select_column:
    :param select_criteria:
    :return:
    """

    assert type(table_name) == str, f'table_name [{table_name}] has to be of str type'
    assert type(select_column) == str, f'select_column [{select_column}] has to be of str type'

    return_df = True
    pd_native = False
    sql_str = f'select {select_column} from {table_name} '
    sql_str = sql_append_criteria(sql_str, select_criteria)
    existing_value = db_select(db_session, sql_str, return_df, pd_native)

    return existing_value


def db_update_single_column(db_session, table_name, select_column, update_value, select_criteria=None):
    """
    :param db_session:
    :param table_name:
    :param select_column:
    :param update_value:
    :param select_criteria:
    :return:
    """

    assert type(table_name) == str, f'table_name [{table_name}] has to be of str type'
    assert type(select_column) == str, f'select_column [{select_column}] has to be of str type'

    sql_str = f'update {table_name} '
    sql_str += f'set {select_column} = {update_value} '
    sql_str = sql_append_criteria(sql_str, select_criteria)
    db_execute(db_session, sql_str)


def db_update_single_value(db_session, table_name, select_column, update_value, select_criteria=None):
    """
    :param db_session:
    :param table_name:
    :param select_column:
    :param update_value:
    :param select_criteria:
    :return:
    """

    existing_value = db_select_by_criteria(db_session, table_name, select_column, select_criteria)
    if not existing_value.empty:
        assert len(existing_value.index) == 1, \
            'existing db value has to be a single value, please check select_criteria'

    # make sure db value type and new value type align
    existing_element = existing_value.ix[0, select_column]
    existing_type = type(existing_element)
    update_type = type(update_value)
    if existing_type != update_type:
        converted_update_value = str_converter(in_str=update_value, out_type=existing_type.__name__)

    if existing_element != converted_update_value:
        db_update_single_column(db_session, table_name, select_column, update_value, select_criteria)
