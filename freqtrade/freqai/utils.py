import copy
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import rapidjson
from PIL import Image
from wordcloud import ImageColorGenerator, WordCloud

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history.history_utils import refresh_backtest_ohlcv_data
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.exchange.exchange import market_is_active
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.plot.plotting import go, make_subplots, store_plot_file
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist


logger = logging.getLogger(__name__)


def download_all_data_for_training(dp: DataProvider, config: Config) -> None:
    """
    Called only once upon start of bot to download the necessary data for
    populating indicators and training the model.
    :param timerange: TimeRange = The full data timerange for populating the indicators
                                    and training the model.
    :param dp: DataProvider instance attached to the strategy
    """

    if dp._exchange is None:
        raise OperationalException('No exchange object found.')
    markets = [p for p, m in dp._exchange.markets.items() if market_is_active(m)
               or config.get('include_inactive')]

    all_pairs = dynamic_expand_pairlist(config, markets)

    timerange = get_required_data_timerange(config)

    new_pairs_days = int((timerange.stopts - timerange.startts) / 86400)

    refresh_backtest_ohlcv_data(
        dp._exchange,
        pairs=all_pairs,
        timeframes=config["freqai"]["feature_parameters"].get("include_timeframes"),
        datadir=config["datadir"],
        timerange=timerange,
        new_pairs_days=new_pairs_days,
        erase=False,
        data_format=config.get("dataformat_ohlcv", "json"),
        trading_mode=config.get("trading_mode", "spot"),
        prepend=config.get("prepend_data", False),
    )


def get_required_data_timerange(config: Config) -> TimeRange:
    """
    Used to compute the required data download time range
    for auto data-download in FreqAI
    """
    time = datetime.now(tz=timezone.utc).timestamp()

    timeframes = config["freqai"]["feature_parameters"].get("include_timeframes")

    max_tf_seconds = 0
    for tf in timeframes:
        secs = timeframe_to_seconds(tf)
        if secs > max_tf_seconds:
            max_tf_seconds = secs

    startup_candles = config.get('startup_candle_count', 0)
    indicator_periods = config["freqai"]["feature_parameters"]["indicator_periods_candles"]

    # factor the max_period as a factor of safety.
    max_period = int(max(startup_candles, max(indicator_periods)) * 1.5)
    config['startup_candle_count'] = max_period
    logger.info(f'FreqAI auto-downloader using {max_period} startup candles.')

    additional_seconds = max_period * max_tf_seconds

    startts = int(
        time
        - config["freqai"].get("train_period_days", 0) * 86400
        - additional_seconds
    )
    stopts = int(time)
    data_load_timerange = TimeRange('date', 'date', startts, stopts)

    return data_load_timerange


# Keep below for when we wish to download heterogeneously lengthed data for FreqAI.
# def download_all_data_for_training(dp: DataProvider, config: Config) -> None:
#     """
#     Called only once upon start of bot to download the necessary data for
#     populating indicators and training a FreqAI model.
#     :param timerange: TimeRange = The full data timerange for populating the indicators
#                                     and training the model.
#     :param dp: DataProvider instance attached to the strategy
#     """

#     if dp._exchange is not None:
#         markets = [p for p, m in dp._exchange.markets.items() if market_is_active(m)
#                    or config.get('include_inactive')]
#     else:
#         # This should not occur:
#         raise OperationalException('No exchange object found.')

#     all_pairs = dynamic_expand_pairlist(config, markets)

#     if not dp._exchange:
#         # Not realistic - this is only called in live mode.
#         raise OperationalException("Dataprovider did not have an exchange attached.")

#     time = datetime.now(tz=timezone.utc).timestamp()

#     for tf in config["freqai"]["feature_parameters"].get("include_timeframes"):
#         timerange = TimeRange()
#         timerange.startts = int(time)
#         timerange.stopts = int(time)
#         startup_candles = dp.get_required_startup(str(tf))
#         tf_seconds = timeframe_to_seconds(str(tf))
#         timerange.subtract_start(tf_seconds * startup_candles)
#         new_pairs_days = int((timerange.stopts - timerange.startts) / 86400)
#         # FIXME: now that we are looping on `refresh_backtest_ohlcv_data`, the function
#         # redownloads the funding rate for each pair.
#         refresh_backtest_ohlcv_data(
#             dp._exchange,
#             pairs=all_pairs,
#             timeframes=[tf],
#             datadir=config["datadir"],
#             timerange=timerange,
#             new_pairs_days=new_pairs_days,
#             erase=False,
#             data_format=config.get("dataformat_ohlcv", "json"),
#             trading_mode=config.get("trading_mode", "spot"),
#             prepend=config.get("prepend_data", False),
#         )

# flake8: noqa: C901
def plot_feature_importance(model: Any, pair: str, dk: FreqaiDataKitchen,
                            count_max: int = 200) -> None:
    """
        Plot Best and worst features by importance for a single sub-train.
        :param model: Any = A model which was `fit` using a common library
                            such as catboost or lightgbm
        :param pair: str = pair e.g. BTC/USD
        :param dk: FreqaiDataKitchen = non-persistent data container for current coin/loop
        :param count_max: int = the amount of features to be loaded per column
    """
    from freqtrade.plot.plotting import go, make_subplots, store_plot_file

    # Extract feature importance from model
    logger.info(f'About to plot {pair}')
    models = {}
    if 'FreqaiMultiOutputRegressor' in str(model.__class__):
        for estimator, label in zip(model.estimators_, dk.label_list):
            models[label] = estimator
    else:
        models[dk.label_list[0]] = model

    for label in models:
        mdl = models[label]
        if "catboost.core" in str(mdl.__class__):
            feature_importance = mdl.get_feature_importance()
        elif "lightgbm.sklearn" or "xgb" in str(mdl.__class__):
            feature_importance = mdl.feature_importances_
        else:
            logger.info('Model type not support for generating feature importances.')
            return

        # Data preparation
        fi_df = pd.DataFrame({
            "feature_names": np.array(dk.data_dictionary['train_features'].columns),
            "feature_importance": np.array(feature_importance)
        })
        fi_df_top = fi_df.nlargest(count_max, "feature_importance")[::-1]
        fi_df_worst = fi_df.nsmallest(count_max, "feature_importance")[::-1]

        # Plotting
        def add_feature_trace(fig, fi_df, col):
            return fig.add_trace(
                go.Bar(
                    x=fi_df["feature_importance"],
                    y=fi_df["feature_names"],
                    orientation='h', showlegend=False
                ), row=1, col=col
            )
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.5)
        fig = add_feature_trace(fig, fi_df_top, 1)
        fig = add_feature_trace(fig, fi_df_worst, 2)
        fig.update_layout(title_text=f"Best and worst features by importance {pair}")
        label = label.replace('&', '').replace('%', '')  # escape two FreqAI specific characters
        store_plot_file(fig, f"{dk.model_filename}-{label}.html", dk.data_path)

        try:
            # save latest feature importances
            df_feature_importances = make_feature_importance_df(
                fi_df, dk.data_path, dk.model_filename, dk.freqai_config)
            filename = f'{dk.data_path}/{dk.model_filename}-{label}-feature-importances.pkl'
            df_feature_importances.to_pickle(filename)
            logger.info(f"Stored feature importances as {filename}")
        except Exception as e:
            logger.warning("Something went wrong trying to pickle feature"
                           f"importances {e}")

        # Plot wordcloud
        train_vals = dk.freqai_config["model_training_parameters"].values()
        try:
            if "gpu_hist" in train_vals or "GPU" in train_vals:
                # only do wordclouds for high performance runs
                paths = dk.freqai_config.get("word_cloud_mask_paths",
                                             ["user_data/plot/word_cloud_mask.jpg"])
                img_path = random.choice(paths)
                cloud = create_wordcloud(fi_df=fi_df, img_path=img_path)
                filename = f'{dk.data_path}/{dk.model_filename}-{label}-wordcloud.png'
                cloud.save(filename, 'PNG')
                logger.info(f"Stored plot as {filename}")
        except Exception as e:
            logger.exception(f"Something went wrong making the word cloud for {pair}, {e}")

        if dk.freqai_config["feature_parameters"]["principal_component_analysis"]:
            plot_pca_correlation(pair, dk)


def make_feature_importance_df(
        fi_df: pd.DataFrame, data_path: Path, model_filename: str, config: Dict
) -> pd.DataFrame:
    """
    Create df containing current and historic (time window of
    config["feature_parameters"]["feature_importance_window_days"]) to be saved as pkl
    for creating dashboard figures.
    """
    df = fi_df.copy()
    pair = model_filename.split('_')[1].upper()
    time_point = datetime.fromtimestamp(
        int(model_filename.split('_')[-1].split('-')[0]),
        tz=timezone.utc
        )
    df_usdt = df.loc[df["feature_names"].str.contains(
        'USDTUSDT')].copy()
    df_usdt["feature_names"] = df_usdt["feature_names"].apply(
        lambda s: s.replace('/USDTUSDT', ''))
    df.loc[df_usdt.index, 'feature_names'] = df_usdt["feature_names"]
    df_santiment = df.loc[df["feature_names"].str.contains(
        '%%-')].copy()
    df_santiment["feature_names"] = df_santiment["feature_names"].apply(
        lambda s: s.replace('%%-', 'S '))
    df.loc[df_santiment.index, 'feature_names'] = df_santiment["feature_names"]
    df_not_santiment = df.loc[df["feature_names"].str.contains(
        '%-')].copy()
    df_not_santiment["feature_names"] = df_not_santiment["feature_names"].apply(
        lambda s: s.replace('%-', ''))
    df.loc[df_not_santiment.index, 'feature_names'] = df_not_santiment["feature_names"]
    df["feature_names"] = df["feature_names"].apply(
        lambda s: s.replace('_', ' '))

    df = df.set_index('feature_names').rename(
        columns={'feature_importance': time_point})

    main_path = data_path.parent
    folders = sorted([f for f in list(main_path.iterdir()) if pair in str(f)], reverse=True)

    if len(folders) == 1:
        return df
    elif len(folders) > 1:
        folder = folders[1]
        file = [f for f in list(folder.iterdir()) if 'feature-importances.pkl' in str(f)]
        if len(file) == 0:
            logger.info('No feature importances .pkl exists in folder %s' % (folder))
            return df
        else:
            df_previous = pd.read_pickle(file[0])
            window = config["feature_parameters"]["feature_importance_window_days"]
            oldest_time_point = df_previous.columns[0]
            if oldest_time_point < df.columns[0] - timedelta(days=window):
                df_previous = df_previous.iloc[:, 1:]
            return pd.concat([df_previous, df], axis=1)


def create_wordcloud(fi_df: pd.DataFrame, img_path: str) -> Image:
    """
    Create a word cloud for feature importances
    """
    # santiment_features = [f for f in fi_df['feature_names'] if '%%-' in f]

    fi_df_santiment = fi_df  # .loc[fi_df["feature_names"].isin(santiment_features)].copy()

    fi_df_santiment["feature_names"] = fi_df_santiment["feature_names"].apply(
        lambda s: s.lstrip('%-'))
    fi_df_santiment["feature_names"] = fi_df_santiment["feature_names"].apply(
        lambda s: s.replace('_', ' '))

    fi_df_shifts = fi_df_santiment.loc[fi_df_santiment["feature_names"].str.contains(
        'shift')].copy()
    fi_df_shifts["feature_names"] = fi_df_shifts["feature_names"].apply(
        lambda s: s.split('shift')[0] + '(shift' + s.split('shift')[1] + ')')
    fi_df_santiment.loc[fi_df_shifts.index, 'feature_names'] = fi_df_shifts["feature_names"]

    # fi_df_eth = fi_df_santiment.loc[fi_df_santiment["feature_names"].str.contains(
    #     'ethereum')].copy()
    # fi_df_eth["feature_names"] = fi_df_eth["feature_names"].apply(
    #     lambda s: s.replace('ethereum', 'ETH'))
    # fi_df_santiment.loc[fi_df_eth.index, 'feature_names'] = fi_df_eth["feature_names"]

    # fi_df_btc = fi_df_santiment.loc[fi_df_santiment["feature_names"].str.contains(
    #     'bitcoin')].copy()
    # fi_df_btc["feature_names"] = fi_df_btc["feature_names"].apply(
    #     lambda s: s.replace('bitcoin', 'BTC'))
    # fi_df_santiment.loc[fi_df_btc.index, 'feature_names'] = fi_df_btc["feature_names"]

    mask = np.array(Image.open(img_path))
    image_colors = ImageColorGenerator(mask)
    wordcloud = WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGBA",
                          mask=mask, max_font_size=100,
                          random_state=42, width=200, height=200)

    d = {a: x for a, x in fi_df_santiment.values}
    return wordcloud.fit_words(d).recolor(color_func=image_colors).to_image()


def record_params(config: Dict[str, Any], full_path: Path) -> None:
    """
    Records run params in the full path for reproducibility
    """
    params_record_path = full_path / "run_params.json"

    run_params = {
        "freqai": config.get('freqai', {}),
        "timeframe": config.get('timeframe'),
        "stake_amount": config.get('stake_amount'),
        "stake_currency": config.get('stake_currency'),
        "max_open_trades": config.get('max_open_trades'),
        "pairs": config.get('exchange', {}).get('pair_whitelist')
    }

    with open(params_record_path, "w") as handle:
        rapidjson.dump(
            run_params,
            handle,
            indent=4,
            default=str,
            number_mode=rapidjson.NM_NATIVE | rapidjson.NM_NAN
        )


def get_timerange_backtest_live_models(config: Config) -> str:
    """
    Returns a formated timerange for backtest live/ready models
    :param config: Configuration dictionary

    :return: a string timerange (format example: '20220801-20220822')
    """
    dk = FreqaiDataKitchen(config)
    models_path = dk.get_full_models_path(config)
    dd = FreqaiDataDrawer(models_path, config)
    timerange = dd.get_timerange_from_live_historic_predictions()
    return timerange.timerange_str


class PerformanceTracker:
    """
    Calculate an accuracy metric based on accuracy scores from matching predictions to targets.
    :params:
    :config: user configuration file
    :pair: coin pair e.g. BTC/USD
    :dd: data drawer
    :dk: data kitchen for current pair
    :do_plot: plot accuracy metric
    """

    def __init__(self, config: Config, pair: str, dd: FreqaiDataDrawer,
                 dk: FreqaiDataKitchen, do_plot: bool = False, verbosity: int = 1):
        self.dk = dk
        self.dd = dd
        self.config = config
        self.pair = pair
        self.verbosity = verbosity
        self.label_period_candles = config["freqai"]["feature_parameters"].get(
            "label_period_candles", 0)

        self.historic_predictions: pd.DataFrame = pd.DataFrame()
        self.compute_model_performance(do_plot)

    def compute_model_performance(self, do_plot: bool):
        from datetime import datetime

        if self.pair not in self.dd.historic_predictions:
            logger.info(f'{self.pair} not yet in historic predictions. No accuracy to track yet.')
            return
        kernel = self.label_period_candles
        num_candles = self.config["freqai"].get("fit_live_predictions_candles", 600) + kernel * 2

        self.historic_predictions = self.dd.historic_predictions[self.pair]
        df_pred_targ = self.create_pred_targ_df()
        self.historic_predictions[
            ['prediction_min', 'target_min', 'prediction_max', 'target_max']
        ] = df_pred_targ[['prediction_min', 'target_min', 'prediction_max', 'target_max']]

        self.historic_predictions = self.historic_predictions.tail(num_candles)
        self.historic_predictions.reset_index(inplace=True)

        t_start = datetime.now()
        accuracy = self.get_accuracy_metrics()
        t_end = np.round((datetime.now() - t_start).total_seconds() * 1000, 2)

        if bool(accuracy):
            self.dd.historic_predictions[self.pair][
                'balanced_accuracy'].iloc[-1] = accuracy['balanced_accuracy']
            self.dd.historic_predictions[self.pair][
                'accuracy_score'].iloc[-1] = accuracy['shift_accuracy']
            curr_acc = np.round(accuracy['balanced_accuracy'], 2)
            logger.debug(f'Current balanced accuracy: {curr_acc}')
            logger.debug(f"Accuracy metric computation took {t_end} ms.")
            if do_plot:
                self.plot_accuracy_metric()

        return

    def get_accuracy_metrics(self) -> Dict:
        """
        Calculate accuracy metrics: accuracy score based on matching predictions and targets (how
        many shifted candles differ between matches), and a balanced accuracy score for the current
        state of the modeling progress.
        :returns:
        :df_metric: dataframe containing accuracy metric and match and price scores
        """
        df_historic_predictions = self.historic_predictions
        num_candles = self.config["freqai"].get("fit_live_predictions_candles", 600)

        warmed_up = df_historic_predictions[
            df_historic_predictions['&s-minima_sort_threshold'] > -2
        ]

        if not warmed_up.shape[0]:
            logger.info(
                f'Warmup is not done. '
                f'No predictions for {self.pair} to calculate accuracy for.'
            )
            accuracy: Dict = {}

            return accuracy

        warmed_up_idx = warmed_up.index[0]

        df_accuracy = self.match_preds_to_targs()

        if df_accuracy[['prediction_min', 'prediction_max']].sum().sum():
            df_metric = self.get_accuracy_score(df_accuracy=df_accuracy)
            shift_accuracy = df_metric.loc[df_metric.index[-1], 'accuracy_score']
            date_shift_accuracy = df_historic_predictions.loc[df_metric.index[-1], 'date_pred']
        else:
            logger.info(
                f'Warmup is done but there are no predicted extrema '
                f'for {self.pair} to calculate accuracy for.'
            )
            shift_accuracy = np.nan
            date_shift_accuracy = df_historic_predictions.loc[
                df_historic_predictions.index[-1], 'date_pred'
            ]

        available_candles = df_historic_predictions.iloc[warmed_up_idx:].shape[0]

        if available_candles < num_candles:
            logger.info(
                f'Waiting on {num_candles - available_candles} candles '
                f'Before accuracy can be computed for {self.pair}.'
            )
            accuracy = {}

            return accuracy

        start_candle = warmed_up_idx + self.label_period_candles
        df_hist = df_historic_predictions.iloc[start_candle: -self.label_period_candles]

        if df_hist.empty:
            logger.warning(
                f'df_hist empty...problem in '
                f' {self.pair}. Len df_historic_predictions {len(df_historic_predictions)}.'
            )
            accuracy = {}

            return accuracy

        df_min = df_hist[['prediction_min', 'target_min']]
        matches = np.where(df_min['prediction_min'] == df_min['target_min'])[0]
        true_positives_min = (df_min.iloc[matches].sum(axis=1) == 2).sum()
        true_negatives_min = (df_min.iloc[matches].sum(axis=1) == 0).sum()
        not_matches = np.where(df_min['prediction_min'] != df_min['target_min'])[0]
        false_positives_min = (df_min['prediction_min'].iloc[not_matches]).sum()

        df_max = df_hist[['prediction_max', 'target_max']]
        matches = np.where(df_max['prediction_max'] == df_max['target_max'])[0]
        true_positives_max = (df_max.iloc[matches].sum(axis=1) == 2).sum()
        true_negatives_max = (df_max.iloc[matches].sum(axis=1) == 0).sum()
        not_matches = np.where(df_max['prediction_max'] != df_max['target_max'])[0]
        false_positives_max = (df_max['prediction_max'].iloc[not_matches]).sum()

        total_positives = df_hist[['target_min', 'target_max']].sum().sum()

        total_negatives = true_negatives_min + true_negatives_max + \
            false_positives_min + false_positives_max

        true_positives = true_positives_min + true_positives_max

        true_negatives = true_negatives_min + true_negatives_max

        sensitivity = true_positives / total_positives
        if not sensitivity == sensitivity:
            sensitivity = 0

        specificity = true_negatives / total_negatives
        if not specificity == specificity:
            specificity = 0

        balanced_accuracy = (sensitivity + specificity) / 2

        if balanced_accuracy < 0:
            logger.warning(f"Warning: the balanced accuracy is negative.\n"
                           f"Specificity: {specificity}\n"
                           f"Sensitivity: {sensitivity}"
                           )

        accuracy = {'balanced_accuracy': balanced_accuracy,
                    'date_balanced_accuracy': df_hist['date_pred'].iloc[-1],
                    'date_shift_accuracy': date_shift_accuracy,
                    'shift_accuracy': shift_accuracy}

        return accuracy  # df_metric.sort_index()

    def get_accuracy_score(self, df_accuracy: pd.DataFrame):
        """
        Calculate an accuracy metric based on accuracy scores from matching predictions to targets.
        :params:
        :df_accuracy: dataframe containing accuracy scores for matched predictions-targets
        :returns:
        :df_metric: dataframe containing accuracy metric and match and price scores
        """

        df_metric = pd.DataFrame()
        df_metric['prediction_idx'] = \
            df_accuracy['prediction_min_idx'].fillna(df_accuracy['prediction_max_idx'])
        df_metric['price_diff'] = df_accuracy['target_close'] - df_accuracy['close_price']
        df_metric['price_score'] = 1 - \
            (df_accuracy['close_price'] - df_accuracy['target_close']).abs() \
            / (df_accuracy['close_price'].max() - df_accuracy['close_price'].min())

        df_metric['min_shift_score'] = 1 - \
            (df_accuracy['prediction_min_idx'] - df_accuracy['target_min_idx']).abs() /\
            (2 * self.label_period_candles)
        df_metric['max_shift_score'] = 1 - \
            (df_accuracy['prediction_max_idx'] - df_accuracy['target_max_idx']).abs() /\
            (2 * self.label_period_candles)
        df_metric['match_score'] = \
            df_metric['min_shift_score'].fillna(df_metric['max_shift_score'])
        df_metric['accuracy_score'] = df_metric['match_score']
        df_metric.dropna(axis=0, how='all', inplace=True)
        df_metric['date'] = df_accuracy['date']

        df_metric = df_metric[['accuracy_score', 'match_score', 'price_score',
                               'prediction_idx', 'date']]
        df_metric = df_metric.set_index(df_metric['prediction_idx'].astype(int))
        df_metric.drop('prediction_idx', axis=1, inplace=True)

        return df_metric

    def match_preds_to_targs(self):
        """
        Match predictions to targets and get distances between matches, and corresponding close
        prices.
        :returns:
        :df_accuracy: dataframe containing the matched predictions-targets with distances between
        each prediction and its closest target, as well as the target indices, and close prices.
        """
        from scipy.spatial.distance import cdist

        df_accuracy = copy.deepcopy(self.historic_predictions[
            ['prediction_min', 'target_min', 'prediction_max', 'target_max']
        ])

        df_accuracy['close_price'] = self.historic_predictions['close_price']
        df_accuracy['date'] = self.historic_predictions['date_pred']
        idx = df_accuracy[df_accuracy['prediction_min'] != 0].index
        df_accuracy.loc[idx, 'prediction_min_idx'] = idx.astype(int)
        idx = df_accuracy[df_accuracy['prediction_max'] != 0].index
        df_accuracy.loc[idx, 'prediction_max_idx'] = idx.astype(int)

        # remove where neither pred nor targ says there is a max/min
        ind_neither = np.where(
            (df_accuracy['target_min'] == 0) & (df_accuracy['prediction_min'] == 0) &
            (df_accuracy['target_max'] == 0) & (df_accuracy['prediction_max'] == 0))[0]
        drop_idx = df_accuracy.iloc[ind_neither].index
        df_accuracy = df_accuracy.drop(drop_idx, axis=0)
        # find distances for minima
        df_match_min = df_accuracy[['prediction_min', 'target_min']]
        ind_neither = np.where(
            (df_match_min['target_min'] == 0) & (df_match_min['prediction_min'] == 0))[0]
        df_match_min = df_match_min.drop(df_match_min.iloc[ind_neither].index, axis=0)
        df_match_min['index'] = df_match_min.index
        df_dist = pd.DataFrame(
            data=cdist(df_match_min[['index']], df_match_min[['index']]),
            index=df_match_min.index, columns=df_match_min.index
        )
        df_dist.iloc[df_match_min['target_min'].eq(0)] = np.nan
        df_dist.iloc[:, df_match_min['prediction_min'].eq(0)] = np.nan
        idx_min = df_dist.idxmin(axis=0).dropna().astype(int)
        if len(idx_min):
            df_accuracy.loc[idx_min.index, 'shift'] = df_dist.min(axis=0)
            df_accuracy.loc[idx_min.index, 'target_min_idx'] = idx_min
            df_accuracy.loc[idx_min.index, 'target_close'] = \
                self.historic_predictions.loc[idx_min]['close_price'].values
        else:
            df_accuracy[['shift', 'target_min_idx', 'target_close']] = np.NaN
        # find distances for maxima
        df_match_max = df_accuracy[['prediction_max', 'target_max']]
        ind_neither = np.where(
            (df_match_max['target_max'] == 0) & (df_match_max['prediction_max'] == 0))[0]
        df_match_max = df_match_max.drop(df_match_max.iloc[ind_neither].index, axis=0)
        df_match_max['index'] = df_match_max.index
        df_dist = pd.DataFrame(
            data=cdist(df_match_max[['index']], df_match_max[['index']]),
            index=df_match_max.index, columns=df_match_max.index
        )
        df_dist.iloc[df_match_max['target_max'].eq(0)] = np.nan
        df_dist.iloc[:, df_match_max['prediction_max'].eq(0)] = np.nan
        idx_max = df_dist.idxmin(axis=0).dropna().astype(int)
        if len(idx_max):
            df_accuracy.loc[idx_max.index, 'shift'] = df_dist.min(axis=0)
            df_accuracy.loc[idx_max.index, 'target_max_idx'] = idx_max
            df_accuracy.loc[idx_max.index, 'target_close'] = \
                self.historic_predictions.loc[idx_max]['close_price'].values
        else:
            df_accuracy[['target_max_idx']] = np.NaN
        if self.verbosity:
            # totals = self.df_pred_targ.abs().sum(axis=0)
            totals = self.historic_predictions[
                ['prediction_min', 'target_min', 'prediction_max', 'target_max']
                ].abs().sum(axis=0)
            identified = df_accuracy[
                ['prediction_min', 'target_min', 'prediction_max', 'target_max']
            ].sum(axis=0)
            logger.info(f'Nmb identified predictions: '
                        f'{identified[0]+identified[2]} / {totals[0]+totals[2]}')
            logger.info(f'Nmb identified targets: '
                        f'{identified[1]+identified[3]} / {totals[1]+totals[3]}')

        return df_accuracy.reset_index(drop=True)

    def create_pred_targ_df(self) -> pd.DataFrame:
        """
        Create dataframe containing predictions and targets.
        :returns:
        :df_pred_targ: dataframe containing predictions and targets
        """
        df_pred_targ = pd.DataFrame(
            index=self.historic_predictions.index,
            columns=['prediction_min', 'target_min', 'prediction_max', 'target_max']
        )
        df_pred_targ = self.identify_predictions(df_pred_targ)
        df_pred_targ = self.identify_targets(df_pred_targ)
        df_pred_targ['target_min'] = np.where(df_pred_targ['target_min'].isnull(), 0, 1)
        df_pred_targ['target_max'] = np.where(df_pred_targ['target_max'].isnull(), 0, 1)

        return df_pred_targ

    def identify_predictions(self, df_pred_targ: pd.DataFrame) -> pd.DataFrame:
        """
        Determine predictions based on thresholds.
        :params:
        :df_pred_targ: empty data frame with the same indices as self.historic_predictions,
        and column labels 'prediction_min', 'target_min', 'prediction_max', 'target_max'
        :returns:
        :df_pred_targ: 'prediction_min', 'prediction_max' containing categorical predictions
        """
        df_pred_targ['prediction_max'] = np.where(
            self.historic_predictions['&s-extrema'] >
            self.historic_predictions['&s-maxima_sort_threshold'],
            1, 0
        )
        df_pred_targ['prediction_min'] = np.where(
            self.historic_predictions['&s-extrema'] <
            self.historic_predictions['&s-minima_sort_threshold'],
            1, 0
        )

        return df_pred_targ

    def identify_targets(self, df_pred_targ: pd.DataFrame) -> pd.DataFrame:
        """
        Identify targets as maxima and minima in close price.
        :params:
        :df_pred_targ: empty data frame with the same indices as self.historic_predictions,
        and column labels 'prediction_min', 'target_min', 'prediction_max', 'target_max'
        :returns:
        :df_pred_targ: 'target_min', 'target_max' containing categorical predictions
        """
        from scipy.signal import argrelextrema

        target_min_idx = argrelextrema(
            self.historic_predictions['close_price'].values, np.less,
            order=self.label_period_candles)[0]
        df_pred_targ['target_min'].iloc[target_min_idx] = 1
        target_max_idx = argrelextrema(
            self.historic_predictions['close_price'].values, np.greater,
            order=self.label_period_candles)[0]
        df_pred_targ['target_max'].iloc[target_max_idx] = 1

        return df_pred_targ

    def plot_predictions_targets(self, fig: make_subplots) -> make_subplots:
        """
        Plot close price chart with prediction and target maxima and minima.
        :params:
        :fig: plotly subplot axis
        :returns:
        :fig: plotted close price chart with prediction and target maxima and minima
        """
        # def add_feature_trace(fig, fi_df_price, fi_df_locs, label, colors, price_offset):
        def add_feature_trace(fig, fi_df, label, colors, price_offset):
            text_offset = 10
            text = ''.join((label[0].upper(), label[1:], 's'))
            fig.add_trace(
                go.Scatter(
                    mode='lines',
                    x=fi_df['date_pred'],
                    y=fi_df['close_price'] * price_offset,
                    name=f'Offset x{price_offset}',
                    line=dict(color='rgb(127, 127, 127, 0.5)'),
                    showlegend=False,
                ),
                secondary_y=False,
            )
            identifiers = ['min', 'max']
            markers = ['triangle-down', 'triangle-up']
            for i in range(2):
                fig.add_trace(
                    go.Scatter(
                        mode='markers',
                        x=fi_df.loc[
                            fi_df[fi_df[f'{label}_{identifiers[i]}'].eq(1)].index
                        ]['date_pred'],
                        y=fi_df.loc[
                            fi_df[fi_df[f'{label}_{identifiers[i]}'].eq(1)].index
                        ]['close_price'] * price_offset,
                        name=''.join((text[:4], f'. {identifiers[i]}ima')),
                        marker=dict(color=colors[i], symbol=markers[i], size=12,
                                    line=dict(width=2, color='black'))
                    ),
                    secondary_y=False,
                )
            fig.add_annotation(
                dict(
                    text=text,
                    font=dict(color='rgb(127, 127, 127, 0.5)'),
                    x=fi_df['date_pred'].values[-1] + text_offset,
                    y=fi_df['close_price'].values[-1] * price_offset,
                    showarrow=False,
                    xanchor='left',
                    xref="x",
                    yref="y",
                )
            )
            return fig

        fig = add_feature_trace(fig, self.historic_predictions,
                                label='prediction',
                                colors=['rgb(214, 39, 40)', 'rgb(44, 160, 44)'],
                                price_offset=1)
        fig = add_feature_trace(fig, self.historic_predictions,
                                label='target',
                                colors=['rgb(227, 119, 194)', 'rgb(23, 190, 207)'],
                                price_offset=1.02)

        return fig

    def plot_accuracy_metric(self) -> None:
        """
        Plot matched predictions-targets with accuracy metric.
        :params:
        :df_metric: dataframe containing accuracy scores
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig = self.plot_predictions_targets(fig=fig)

        def add_feature_trace(fig, fi_df, label, colors):
            text = f'Avg. {fi_df.loc[:, label[0]].mean().round(3)}'
            fig.add_trace(
                go.Scatter(
                    x=fi_df['date_pred'],
                    y=fi_df.loc[:, label[0]],
                    showlegend=True,
                    name=label[1],
                    line=dict(color=colors[0]),
                ),
                secondary_y=True,
            )
            fig.add_annotation(
                dict(
                    text=text,
                    font=dict(color=colors[0]),
                    x=fi_df['date_pred'].values[-1],
                    y=fi_df[label[0]].iloc[-1],
                    showarrow=False,
                    xanchor='left',
                    xref="x",
                    yref="y2",
                ),
            )
            return fig

        df_metric = self.historic_predictions

        if not all(df_metric['balanced_accuracy'].isnull()):
            fig = add_feature_trace(
                fig=fig, fi_df=df_metric[['date_pred', 'balanced_accuracy']].ffill(),
                label=['balanced_accuracy', 'Balanced accuracy'],
                colors=['rgb(34, 189, 117)']
            )
        if not all(df_metric['accuracy_score'].isnull()):
            fig = add_feature_trace(
                fig=fig, fi_df=df_metric[['date_pred', 'accuracy_score']].dropna(
                    subset='accuracy_score'),
                label=['accuracy_score', 'Accuracy score'],
                colors=['rgb(214, 39, 40)']
            )

        fig.update_yaxes(
            showgrid=False, title_text="<b>Close price</b>",
            secondary_y=False
        )
        fig.update_yaxes(
            showgrid=True, title_text="<b>Accuracy score</b>",
            secondary_y=True  # , range=[-0.01, 1.01]
        )
        fig.update_layout(title_text=f"<b>Accuracy score for {self.pair}</b>")

        store_plot_file(fig, f"{self.dk.model_filename}_accuracy-score.html", self.dk.data_path)


def plot_pca_correlation(pair: str, dk: FreqaiDataKitchen) -> None:
    """
    Plot the correlation matrix for the nmb_pcs=10 most and least important features
    for the 5 first PCs.
    :params:
    :pair: coin pair
    :dk: non-persistent data container for current coin/loop
    """

    if not dk.pca:
        logger.info('No PCA model exists for generating correlation plots')
        return
    else:
        from freqtrade.plot.plotting import make_subplots, store_plot_file

        nmb_pcs = 10

        correlation_matrix = get_pca_correlation(dk)
        corr_idx = correlation_matrix.index.str.lstrip('%-')
        explained_variance_ratios = dk.pca.explained_variance_ratio_[:nmb_pcs]

        most_important_features, least_important_features = get_important_pca_features(dk)

        most_important_features = list(most_important_features.feature_name[:nmb_pcs])
        most_important_correls = correlation_matrix.filter(
            most_important_features
        ).iloc[:nmb_pcs]
        column_labels = [
            f'{corr_idx[i]}:\n{column_label}' for i, column_label in enumerate(
                most_important_correls.columns.str.lstrip('%-')
            )
        ]
        most_important_correls.columns = column_labels
        most_important_correls = most_important_correls.iloc[::-1]

        least_important_features = list(
            least_important_features.feature_name[:nmb_pcs]
        )
        least_important_correls = correlation_matrix.filter(
            least_important_features).iloc[:nmb_pcs]
        column_labels = [
            f'{corr_idx[i]}:\n{column_label}' for i, column_label in enumerate(
                least_important_correls.columns.str.lstrip('%-')
            )
        ]
        least_important_correls.columns = column_labels
        least_important_correls = least_important_correls.iloc[::-1]

        y_labels = []
        for i in range(nmb_pcs):
            y_labels.append(
                f'{corr_idx[nmb_pcs-i-1]} '
                f'({np.round(explained_variance_ratios[nmb_pcs-i-1]*100, 1)}%)'
            )

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Most important features', 'Least important features')
        )
        add_plot(
            fig=fig, dataframe=most_important_correls,
            x_labels=most_important_correls.columns, y_labels=y_labels,
            col=1, nmb_pcs=nmb_pcs,
        )
        add_plot(
            fig=fig, dataframe=least_important_correls,
            x_labels=least_important_correls.columns, y_labels=y_labels,
            col=2, nmb_pcs=nmb_pcs,
        )

        for i in range(nmb_pcs):
            fig.add_shape(
                dict(
                    type='rect', x0=(i - .5), y0=(nmb_pcs - 1.5 - i),
                    x1=(i + .5), y1=(nmb_pcs - i - .5)
                ),
                line=dict(color='black', width=2),
                row="all", col="all",
            )

        fig.update_layout(
            title_text=f'Correlation coefficients'
            f'\nfor the first {nmb_pcs} PCs '
            f'({np.round(dk.pca.explained_variance_ratio_[:nmb_pcs].sum()*100,1)}% '
            f'explained variance) and the corresponding most and least important features'
        )

        store_plot_file(
            fig, f"{dk.model_filename}_correlation-PCA-TrainFeatures.html", dk.data_path)


def add_plot(fig, dataframe, x_labels, y_labels, col, nmb_pcs) -> None:
    from freqtrade.plot.plotting import go
    fig.add_trace(
        go.Heatmap(
            z=dataframe.values, x=x_labels, y=y_labels,
            text=dataframe.round(decimals=2).values,
            texttemplate='%{text}', textfont={"size": 12},
            hoverongaps=False, colorscale='RdBu', reversescale=True,
            zmin=-1, zmax=1, zmid=0,
        ),
        row=1, col=col
    )


def get_pca_correlation(dk: FreqaiDataKitchen) -> pd.DataFrame:
    """
    Get the correlation matrix between principal components and the original features.
    :returns:
    :df_loadings: dataframe containing the component loadings (correlation coefficients)
    for the features in the input dataframe
    """
    if not dk.pca:
        logger.info('No PCA model exists')
        return
    else:
        loadings = dk.pca.components_.T * np.sqrt(dk.pca.explained_variance_)
        loadings = loadings.T
        nmb_pcs = dk.pca.n_components_
        pcs_list = [f'PC{str(i + 1)}' for i in range(nmb_pcs)]
        df_loadings = pd.DataFrame.from_dict(dict(zip(pcs_list, loadings)))
        df_loadings['Feature'] = dk.data['training_features_list_raw']
        df_loadings = df_loadings.set_index('Feature')

        return df_loadings.transpose()


def get_important_pca_features(dk: FreqaiDataKitchen) -> Tuple:
    """
    Get the most and least important features for the principal components
    after PCA transformation.
    :returns:
    :df_most_important: dataframe linking the names of the most inluencial original features
    to the corresponding principal components
    :df_least_important: dataframe linking the names of the least inluencial original features
    to the corresponding principal components
    """
    if not dk.pca:
        logger.info('No PCA model exists')
        return None, None
    else:
        nmb_pcs = dk.pca.n_components_
        feature_names = dk.data['training_features_list_raw']

        most_important_pcs = [np.abs(dk.pca.components_[i]).argmax() for i in range(nmb_pcs)]
        most_important_feature_names = [
            feature_names[most_important_pcs[i]] for i in range(nmb_pcs)
        ]
        dic = {f'PC{str(i + 1)}': most_important_feature_names[i] for i in range(nmb_pcs)}
        df_most_important = pd.DataFrame(
            data=dic.items(), columns=['principal_component', 'feature_name']
        )

        least_important_pcs = [np.abs(dk.pca.components_[i]).argmin() for i in range(nmb_pcs)]
        least_important_feature_names = [
            feature_names[least_important_pcs[i]] for i in range(nmb_pcs)
        ]
        dic = {f'PC{str(i + 1)}': least_important_feature_names[i] for i in range(nmb_pcs)}
        df_least_important = pd.DataFrame(
            data=dic.items(), columns=['principal_component', 'feature_name']
        )

        return df_most_important, df_least_important
