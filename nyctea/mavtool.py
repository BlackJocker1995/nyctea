"""Log-analysis and plotting helpers.

The param loading / scaling / geometry formerly here moved to :mod:`nyctea.params`.
This module keeps the remaining mavtool utilities that are not SITL-coupled:

- :func:`extract_log_file_des_and_ach` — parse a ``.BIN`` log's ATT/RATE
  messages into a desired-vs-achieved attitude DataFrame (for plotting and the
  analyze stage).
- :func:`sort_result_detect_repair` — classify whether an outcome timestamp
  falls before detection, after detection, or after repair.
- :func:`draw_att_des_and_ach` / :func:`draw_att_des_and_ach_repair` — the
  matplotlib figures (English / Chinese variants) used by the analyze stage.
- :func:`send_notice` — the webhook notifier (kept verbatim; token is configured
  per-deployment, remove if unused).

The plotting functions are preserved verbatim from the legacy ``Cptool/mavtool``
so published figures stay byte-for-byte reproducible.
"""
import numpy as np
import pandas as pd
from pymavlink import mavutil

# matplotlib and requests are only needed by the plotting/notify helpers below;
# import them lazily so importing this module (and thus the core package) does
# not hard-require matplotlib in headless / minimal environments.



# --------------------------------------------------------------------- log parse
def extract_log_file_des_and_ach(log_file):
    """Extract desired-vs-achieved attitude from a ``.BIN`` log's ATT/RATE msgs.

    Returns a DataFrame indexed by TimeS (0.1s resolution, deduplicated) with
    the Roll/Pitch/Yaw desired+achieved and rate channels merged.
    """
    logs = mavutil.mavlink_connection(log_file)
    out_data = []

    while True:
        msg = logs.recv_match(type=["ATT", "RATE"])
        if msg is None:
            break
        if msg.get_type() == "ATT":
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'Roll': msg.Roll,
                'DesRoll': msg.DesRoll,
                'Pitch': msg.Pitch,
                'DesPitch': msg.DesPitch,
                'Yaw': msg.Yaw,
                'DesYaw': msg.DesYaw,
            }
        else:
            out = {
                'TimeS': msg.TimeUS / 1000000,
                # deg to rad
                'DesRateRoll': msg.RDes,
                'RateRoll': msg.R,
                'DesRatePitch': msg.PDes,
                'RatePitch': msg.P,
                'DesRateYaw': msg.YDes,
                'RateYaw': msg.Y,
            }
        out_data.append(out)

    pd_array = pd.DataFrame(out_data)
    pd_array['TimeS'] = pd_array['TimeS'].round(1)
    pd_array = pd_array.drop_duplicates(keep='first')
    # merge data in same TimeS
    df_array = pd.DataFrame(columns=pd_array.columns)
    for group, group_item in pd_array.groupby('TimeS'):
        # filling
        group_item = group_item.fillna(method='ffill')
        group_item = group_item.fillna(method='bfill')
        df_array.loc[len(df_array.index)] = group_item.mean()
    df_array = df_array.fillna(method='ffill')
    df_array = df_array.dropna()
    return df_array


# --------------------------------------------------------------------- helpers
def _systematic_sampling(dataMat, number):
    """Down-sample indices for plotting (picks ``number`` roughly-even rows)."""
    length = dataMat.shape[0]
    k = length // number
    out = range(length)
    out_index = out[:length:k]
    return out_index


def sort_result_detect_repair(result_time, detect_time, repair_time):
    """Classify a result timestamp relative to detect/repair timestamps.

    Returns ``"repair"`` if the result is after repair, ``"detect"`` if after
    detection, else ``"miss"``.
    """
    if result_time > repair_time:
        return "repair"
    if result_time > detect_time:
        return "detect"
    return "miss"


# --------------------------------------------------------------------- plotting
def draw_att_des_and_ach(pdarray, exec='pdf'):
    """Plot desired-vs-achieved attitude (English labels)."""
    from matplotlib import pyplot as plt
    index = _systematic_sampling(pdarray, 300)
    pdarray = pdarray.iloc[index]
    for name in ['Roll', 'Pitch', 'Yaw']:
        x = pdarray[name].to_numpy()
        y = pdarray[f"Des{name}"].to_numpy()

        loss = np.abs(x - y)

        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot()

        ax2 = ax1.twinx()

        ax2.fill_betweenx([0, 10 * np.max(np.abs(x - y))], [210, 210],
                          [len(x), len(x)], color="tomato", alpha=0.2, label="Unstable Area")

        ax1.plot(y, '-', label='Achieved', linewidth=2)
        ax1.plot(x, '--', label='Desired', linewidth=2)
        ax1.set_xlabel("Timestamp (0.1 Second)", fontsize=18)
        ax1.set_ylabel(f'{name} (deg)', fontsize=18)

        ax2.bar(np.arange(len(x)), loss, label='Bias', color='tab:cyan')
        ax2.set_ylim([0, 10 * np.max(np.abs(x - y))])
        ax2.set_ylabel('Bias (deg)', fontsize=18)

        fig.legend(loc='upper center', ncol=2, fontsize='18')
        plt.setp(ax1.get_xticklabels(), fontsize=18)
        plt.setp(ax2.get_yticklabels(), fontsize=18)
        plt.setp(ax1.get_yticklabels(), fontsize=18)

        plt.margins(0, 0)
        plt.gcf().subplots_adjust(bottom=0.12)
        plt.show()


def draw_att_des_and_ach_repair(pdarray, exec='pdf'):
    """Plot desired-vs-achieved attitude with a repair-region annotation (CN labels).

    Preserved verbatim (Chinese font + labels) so the published figure is
    reproducible. The ``repair_line`` timestamp marks where the corrected config
    was uploaded.
    """
    index = _systematic_sampling(pdarray, 500)
    pdarray = pdarray.iloc[index]
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    repair_line = 332  # real 332 ; thrust 305

    for name in ['Roll', 'Pitch', 'Yaw']:
        x = pdarray[name].to_numpy()
        y = pdarray[f"Des{name}"].to_numpy()

        loss = np.abs(x - y)

        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot()

        ax2 = ax1.twinx()

        ax2.fill_betweenx([0, 10 * np.max(loss)], [0, 0],
                          [repair_line, repair_line], color="tomato", alpha=0.2, label="不稳定区域")

        ax2.fill_betweenx([0, 10 * np.max(np.abs(x - y))], [repair_line, repair_line],
                          [len(x), len(x)], color="green", alpha=0.2, label="被修复的区域")

        mid = np.sqrt(x.max() - x.min())

        ax1.plot(y, '-', label='实现的', linewidth=2)
        ax1.plot(x, '--', label='期望的', linewidth=2)
        ax1.set_xlabel("时间戳 (0.1 秒)", fontsize=18)
        ax1.set_ylabel(f'{name} (deg)', fontsize=18)
        ax1.annotate('整改上传', xy=(repair_line, x.min() + mid * 0.5),
                     xytext=(repair_line + pdarray.shape[0] * 0.1, x.min() + mid * 0.5),
                     arrowprops=dict(arrowstyle="->", color="r", hatch='*', ), fontsize='16')

        ax2.bar(np.arange(len(x)), loss, label='差距', color='tab:cyan')
        ax2.set_ylim([0, 10 * np.max(np.abs(x - y))])
        ax2.set_ylabel('差距 （deg）', fontsize=18)

        fig.legend(loc='upper center', ncol=3, fontsize='18')
        plt.setp(ax1.get_xticklabels(), fontsize=18)
        plt.setp(ax2.get_yticklabels(), fontsize=18)
        plt.setp(ax1.get_yticklabels(), fontsize=18)

        plt.margins(0, 0)
        plt.gcf().subplots_adjust(bottom=0.12)
        plt.subplots_adjust(left=0.125, bottom=0.132, right=0.88, top=0.79, wspace=0.2, hspace=0.2)
        plt.show()


# --------------------------------------------------------------------- notifier
def send_notice(thread, buffer_len, content):
    """Webhook notifier (legacy). The token is deployment-specific; remove if unused."""
    import requests
    url = (f"http://iyuu.cn/IYUU5945T5e031af7ab34a0248e4ed4318d9c126efd285bd0.send?text="
           f"Nyctea-{thread}错误&desp=Buffer-{buffer_len}-{content}")
    requests.request("GET", url)
