import pandas as pd
from pandas import Timestamp
import datetime
import numpy as np


def make_sample_data_random_ts(n_rows,
                               n_columns,
                               random_seed=None,
                               column_labels=None,
                               use_date_index=True,
                               start_date=None,
                               end_date=None):
    """

    :param random_seed:
    :param column_labels:
    :param use_date_index:
    :param n_rows:
    :param n_columns:
    :param start_date:
    :param end_date:
    usage:
        >>> start_date = datetime.date(2010, 1, 1)
        >>> end_date = datetime.date(2020, 1, 3)
        no start nor end date
        >>> make_sample_data_random_ts(100, 3)

        with start date only
        >>> make_sample_data_random_ts(100, 3, start_date=start_date)

        with end date only
        >>> make_sample_data_random_ts(100, 3, end_date=end_date)

        start date and end date
        >>> make_sample_data_random_ts(100, 3, start_date=start_date, end_date=end_date)
    """
    if use_date_index:
        if start_date is not None and end_date is not None:
            date_index = pd.date_range(start_date, end_date)
            if len(date_index) < n_rows:
                e_date = start_date + datetime.timedelta(days=n_rows)
                date_index = pd.date_range(start_date, e_date)
            else:
                date_index = date_index[:n_rows]
        elif start_date is not None:
            # date_range includes the start date
            date_index = pd.date_range(start_date, start_date + datetime.timedelta(days=n_rows - 1))
        elif end_date is not None:
            date_index = pd.date_range(end_date - datetime.timedelta(days=n_rows - 1), end_date)
        else:
            today_date = datetime.date.today()
            date_index = pd.date_range(today_date - datetime.timedelta(days=n_rows - 1), today_date)

        if random_seed is not None:
            np.random.seed(random_seed)
        result_df = pd.DataFrame(np.random.rand(n_rows, n_columns),
                                 index=date_index)
    else:
        result_df = pd.DataFrame(np.random.rand(n_rows, n_columns))

    if column_labels is not None:
        assert len(column_labels) == n_columns, 'column_labels not provided'
        result_df.columns = column_labels

    return result_df


def make_sample_data_simple():
    test_df = pd.DataFrame(
        {"a": [1, 2, 3, 4],
         "b": [3, 4, 5, 9],
         "c": [4, 6, 8, 13],
         "date": [datetime.datetime(2020, 1, 3), datetime.datetime(2020, 1, 4), datetime.datetime(2020, 1, 5),
                  datetime.datetime(2020, 1, 6)]}
    )
    test_df['date'] = test_df['date'].map(datetime.datetime.timestamp)
    return test_df


def make_sample_data_ts(read_from_bucket=True, file_list=None, n_item_limit=None):
    """
    Returns timeseries sample data, can load it directly from the google storage bucket or
    just directly re-create it from memory

    :param n_item_limit:
    :param read_from_bucket:
    :param file_list:
    :return:
    usage:
    >>> make_sample_data_ts(False)
    """
    if read_from_bucket:
        if file_list is None:
            f_list = ['gs://at-ml-bucket/prices/ib_5secs_trades_spx_20200728_20200729.parquet',
                      'gs://at-ml-bucket/prices/ib_5secs_trades_vxq0_20200728_20200729.parquet']
        else:
            f_list = file_list

        loaded_dict = {}
        for f in f_list:
            df_loaded = pd.read_parquet(f)[['date_time', 'close', 'ticker']].set_index('date_time')
            loaded_dict[df_loaded['ticker'].values[0]] = df_loaded['close']

        combined_df = pd.DataFrame(loaded_dict).dropna().pct_change().dropna()
    else:
        combined_df = pd.DataFrame(
            {'spx': {Timestamp('2020-07-28 09:30:05', freq='5S'): -4.946791078452861e-05,
                     Timestamp('2020-07-28 09:30:10', freq='5S'): 9.584881858604177e-05,
                     Timestamp('2020-07-28 09:30:15', freq='5S'): -4.637401571150335e-05,
                     Timestamp('2020-07-28 09:30:20', freq='5S'): 0.00018241292101839335,
                     Timestamp('2020-07-28 09:30:25', freq='5S'): 0.00024111208311561327,
                     Timestamp('2020-07-28 09:30:30', freq='5S'): 0.00023487309126979383,
                     Timestamp('2020-07-28 09:30:35', freq='5S'): 0.00016993403469744983,
                     Timestamp('2020-07-28 09:30:40', freq='5S'): -0.0005560532575452637,
                     Timestamp('2020-07-28 09:30:45', freq='5S'): -5.8727165950678284e-05,
                     Timestamp('2020-07-28 09:30:50', freq='5S'): 8.036821004542283e-05,
                     Timestamp('2020-07-28 09:30:55', freq='5S'): -0.0001792685226109647,
                     Timestamp('2020-07-28 09:31:00', freq='5S'): 0.00021639735500600388,
                     Timestamp('2020-07-28 09:31:05', freq='5S'): 9.890310277582692e-05,
                     Timestamp('2020-07-28 09:31:10', freq='5S'): 8.344124036940848e-05,
                     Timestamp('2020-07-28 09:31:15', freq='5S'): -0.00021013077550613435,
                     Timestamp('2020-07-28 09:31:20', freq='5S'): 6.181615874378821e-05,
                     Timestamp('2020-07-28 09:31:25', freq='5S'): 8.344665595250156e-05,
                     Timestamp('2020-07-28 09:31:30', freq='5S'): -7.725897517518732e-05,
                     Timestamp('2020-07-28 09:31:35', freq='5S'): -0.00029669738720861094,
                     Timestamp('2020-07-28 09:31:40', freq='5S'): 3.40066653063964e-05,
                     Timestamp('2020-07-28 09:31:45', freq='5S'): -0.0002163986929518691,
                     Timestamp('2020-07-28 09:31:50', freq='5S'): -0.00031848413918067475,
                     Timestamp('2020-07-28 09:31:55', freq='5S'): 0.0001824907826690758,
                     Timestamp('2020-07-28 09:32:00', freq='5S'): -5.2572495925606866e-05,
                     Timestamp('2020-07-28 09:32:05', freq='5S'): -3.711194819167041e-05,
                     Timestamp('2020-07-28 09:32:10', freq='5S'): 2.4742217026307856e-05,
                     Timestamp('2020-07-28 09:32:15', freq='5S'): -4.948320972830711e-05,
                     Timestamp('2020-07-28 09:32:20', freq='5S'): -0.0001113427314847204,
                     Timestamp('2020-07-28 09:32:25', freq='5S'): -4.949116891961314e-05,
                     Timestamp('2020-07-28 09:32:30', freq='5S'): -0.00010826729028623472,
                     Timestamp('2020-07-28 09:32:35', freq='5S'): 7.424846630521564e-05,
                     Timestamp('2020-07-28 09:32:40', freq='5S'): 0.00038977550786811754,
                     Timestamp('2020-07-28 09:32:45', freq='5S'): 0.0003463321263248442,
                     Timestamp('2020-07-28 09:32:50', freq='5S'): 0.00015146784708530703,
                     Timestamp('2020-07-28 09:32:55', freq='5S'): -2.1634986864471628e-05,
                     Timestamp('2020-07-28 09:33:00', freq='5S'): -0.00015762974318711542,
                     Timestamp('2020-07-28 09:33:05', freq='5S'): -9.892052971949283e-05,
                     Timestamp('2020-07-28 09:33:10', freq='5S'): -3.70988684844642e-05,
                     Timestamp('2020-07-28 09:33:15', freq='5S'): -5.874205436429136e-05,
                     Timestamp('2020-07-28 09:33:20', freq='5S'): 0.00014531782864257714,
                     Timestamp('2020-07-28 09:33:25', freq='5S'): -3.091419456169309e-05,
                     Timestamp('2020-07-28 09:33:30', freq='5S'): 3.40066653063964e-05,
                     Timestamp('2020-07-28 09:33:35', freq='5S'): 5.564537818747439e-05,
                     Timestamp('2020-07-28 09:33:40', freq='5S'): 0.00016692684585550843,
                     Timestamp('2020-07-28 09:33:45', freq='5S'): 6.799588315842264e-05,
                     Timestamp('2020-07-28 09:33:50', freq='5S'): -0.00015143507916348664,
                     Timestamp('2020-07-28 09:33:55', freq='5S'): -0.00013600311570760582,
                     Timestamp('2020-07-28 09:34:00', freq='5S'): 6.491940719288714e-05,
                     Timestamp('2020-07-28 09:34:05', freq='5S'): -4.01855956276842e-05,
                     Timestamp('2020-07-28 09:34:10', freq='5S'): 0.00018547943342217543,
                     Timestamp('2020-07-28 09:34:15', freq='5S'): 8.963176797105454e-05,
                     Timestamp('2020-07-28 09:34:20', freq='5S'): -0.00015761415436921222,
                     Timestamp('2020-07-28 09:34:25', freq='5S'): 7.727401985624383e-05,
                     Timestamp('2020-07-28 09:34:30', freq='5S'): -0.00029980003028906577,
                     Timestamp('2020-07-28 09:34:35', freq='5S'): -6.801627443941971e-05,
                     Timestamp('2020-07-28 09:34:40', freq='5S'): 0.00014531737934020406,
                     Timestamp('2020-07-28 09:34:45', freq='5S'): 2.1639869295153602e-05,
                     Timestamp('2020-07-28 09:34:50', freq='5S'): 8.964894708851645e-05,
                     Timestamp('2020-07-28 09:34:55', freq='5S'): 8.036771319841485e-05,
                     Timestamp('2020-07-28 09:35:00', freq='5S'): 0.00011126942965145048,
                     Timestamp('2020-07-28 09:35:05', freq='5S'): 6.799041953176044e-05,
                     Timestamp('2020-07-28 09:35:10', freq='5S'): -6.180527013532888e-05,
                     Timestamp('2020-07-28 09:35:15', freq='5S'): -3.0904545131438255e-05,
                     Timestamp('2020-07-28 09:35:20', freq='5S'): -1.8543300151163677e-05,
                     Timestamp('2020-07-28 09:35:25', freq='5S'): -3.7087288023096576e-05,
                     Timestamp('2020-07-28 09:35:30', freq='5S'): 0.00027507425459516455,
                     Timestamp('2020-07-28 09:35:35', freq='5S'): -0.00011432526464749682,
                     Timestamp('2020-07-28 09:35:40', freq='5S'): -5.871428085824082e-05,
                     Timestamp('2020-07-28 09:35:45', freq='5S'): -0.00027195579482175436,
                     Timestamp('2020-07-28 09:35:50', freq='5S'): -0.00013292364000561108,
                     Timestamp('2020-07-28 09:35:55', freq='5S'): 0.00012984965265205872,
                     Timestamp('2020-07-28 09:36:00', freq='5S'): -4.6368854967338e-05,
                     Timestamp('2020-07-28 09:36:05', freq='5S'): -8.346780924817221e-05,
                     Timestamp('2020-07-28 09:36:10', freq='5S'): -0.00025660764690804694,
                     Timestamp('2020-07-28 09:36:15', freq='5S'): -6.184903886585769e-05,
                     Timestamp('2020-07-28 09:36:20', freq='5S'): 2.1648502542159775e-05,
                     Timestamp('2020-07-28 09:36:25', freq='5S'): -7.422183049599074e-05,
                     Timestamp('2020-07-28 09:36:30', freq='5S'): 2.474244659489422e-05,
                     Timestamp('2020-07-28 09:36:35', freq='5S'): -1.5463646513280693e-05,
                     Timestamp('2020-07-28 09:36:40', freq='5S'): -2.1649439898130396e-05,
                     Timestamp('2020-07-28 09:36:45', freq='5S'): 0.00010206385485900249,
                     Timestamp('2020-07-28 09:36:50', freq='5S'): 0.0002474022761009831,
                     Timestamp('2020-07-28 09:36:55', freq='5S'): 0.00017313875834767956,
                     Timestamp('2020-07-28 09:37:00', freq='5S'): -8.037193659282682e-05,
                     Timestamp('2020-07-28 09:37:05', freq='5S'): 4.6372151977003284e-05,
                     Timestamp('2020-07-28 09:37:10', freq='5S'): -1.2365333786679145e-05,
                     Timestamp('2020-07-28 09:37:15', freq='5S'): -0.0001483858402812066,
                     Timestamp('2020-07-28 09:37:20', freq='5S'): -6.183660912761901e-05,
                     Timestamp('2020-07-28 09:37:25', freq='5S'): -7.730054141297238e-05,
                     Timestamp('2020-07-28 09:37:30', freq='5S'): -0.00019790468415659124,
                     Timestamp('2020-07-28 09:37:35', freq='5S'): -0.0001515507664138216,
                     Timestamp('2020-07-28 09:37:40', freq='5S'): -9.589358904960044e-05,
                     Timestamp('2020-07-28 09:37:45', freq='5S'): 8.352823254265118e-05,
                     Timestamp('2020-07-28 09:37:50', freq='5S'): -7.11477367286717e-05,
                     Timestamp('2020-07-28 09:37:55', freq='5S'): -0.00019180319754485975,
                     Timestamp('2020-07-28 09:38:00', freq='5S'): -0.0004239045008138653,
                     Timestamp('2020-07-28 09:38:05', freq='5S'): -4.024157325965838e-05,
                     Timestamp('2020-07-28 09:38:10', freq='5S'): -9.286890625204158e-05,
                     Timestamp('2020-07-28 09:38:15', freq='5S'): -8.97816139637797e-05,
                     Timestamp('2020-07-28 09:38:20', freq='5S'): -0.0001486173938082258},
             'vxq0': {Timestamp('2020-07-28 09:30:05', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:30:10', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:30:15', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:30:20', freq='5S'): -0.0036231884057971175,
                      Timestamp('2020-07-28 09:30:25', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:30:30', freq='5S'): 0.001818181818181941,
                      Timestamp('2020-07-28 09:30:35', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:30:40', freq='5S'): 0.001814882032667997,
                      Timestamp('2020-07-28 09:30:45', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:30:50', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:30:55', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:31:00', freq='5S'): -0.0018115942028985588,
                      Timestamp('2020-07-28 09:31:05', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:31:10', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:31:15', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:31:20', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:31:25', freq='5S'): -0.001814882032667886,
                      Timestamp('2020-07-28 09:31:30', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:31:35', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:31:40', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:31:45', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:31:50', freq='5S'): 0.001818181818181941,
                      Timestamp('2020-07-28 09:31:55', freq='5S'): -0.001814882032667886,
                      Timestamp('2020-07-28 09:32:00', freq='5S'): 0.001818181818181941,
                      Timestamp('2020-07-28 09:32:05', freq='5S'): -0.001814882032667886,
                      Timestamp('2020-07-28 09:32:10', freq='5S'): 0.001818181818181941,
                      Timestamp('2020-07-28 09:32:15', freq='5S'): 0.001814882032667997,
                      Timestamp('2020-07-28 09:32:20', freq='5S'): -0.0018115942028985588,
                      Timestamp('2020-07-28 09:32:25', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:32:30', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:32:35', freq='5S'): -0.001814882032667886,
                      Timestamp('2020-07-28 09:32:40', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:32:45', freq='5S'): -0.0018181818181818299,
                      Timestamp('2020-07-28 09:32:50', freq='5S'): 0.0018214936247722413,
                      Timestamp('2020-07-28 09:32:55', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:33:00', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:33:05', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:33:10', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:33:15', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:33:20', freq='5S'): -0.0018181818181818299,
                      Timestamp('2020-07-28 09:33:25', freq='5S'): 0.0018214936247722413,
                      Timestamp('2020-07-28 09:33:30', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:33:35', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:33:40', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:33:45', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:33:50', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:33:55', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:34:00', freq='5S'): -0.0018181818181818299,
                      Timestamp('2020-07-28 09:34:05', freq='5S'): 0.0018214936247722413,
                      Timestamp('2020-07-28 09:34:10', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:34:15', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:34:20', freq='5S'): -0.0018181818181818299,
                      Timestamp('2020-07-28 09:34:25', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:34:30', freq='5S'): 0.0018214936247722413,
                      Timestamp('2020-07-28 09:34:35', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:34:40', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:34:45', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:34:50', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:34:55', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:00', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:05', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:10', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:15', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:20', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:25', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:30', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:35', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:40', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:45', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:50', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:35:55', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:36:00', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:36:05', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:36:10', freq='5S'): 0.001818181818181941,
                      Timestamp('2020-07-28 09:36:15', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:36:20', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:36:25', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:36:30', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:36:35', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:36:40', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:36:45', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:36:50', freq='5S'): -0.001814882032667886,
                      Timestamp('2020-07-28 09:36:55', freq='5S'): -0.0018181818181818299,
                      Timestamp('2020-07-28 09:37:00', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:37:05', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:37:10', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:37:15', freq='5S'): 0.0018214936247722413,
                      Timestamp('2020-07-28 09:37:20', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:37:25', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:37:30', freq='5S'): 0.001818181818181941,
                      Timestamp('2020-07-28 09:37:35', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:37:40', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:37:45', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:37:50', freq='5S'): 0.001814882032667997,
                      Timestamp('2020-07-28 09:37:55', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:38:00', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:38:05', freq='5S'): 0.0018115942028984477,
                      Timestamp('2020-07-28 09:38:10', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:38:15', freq='5S'): 0.0,
                      Timestamp('2020-07-28 09:38:20', freq='5S'): 0.001808318264014508}}
        )
    if n_item_limit is not None:
        combined_df = combined_df.head(n_item_limit)
    return combined_df
