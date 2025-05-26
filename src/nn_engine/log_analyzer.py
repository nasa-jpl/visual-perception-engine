import os
import re
from typing import Optional
from datetime import datetime

import pandas as pd

pd.options.plotting.backend = "plotly"
import numpy as np
import plotly.express as px

from utils.logging_utils import MESSAGE


class LogAnalyzer:
    def __init__(
        self,
        logging_config: dict,
        input_proc: str = "ExampleInput-4",
        fm_proc: str = "FoundationModel-2",
        main_proc: str = "MainProcess",
        heads_proc: list[str] = ["ModelHead-3"],
        output_file: Optional[str] = "",
        verbose: bool = False,
    ) -> None:
        self.log_file = logging_config["log_file"]
        self.output_file = output_file
        self.verbose = verbose
        self.input_proc = input_proc
        self.fm_proc = fm_proc
        self.main_proc = main_proc
        self.heads_proc = heads_proc

        self.metrics = {
            "total_time": ("ADDED_TO_QUEUE", self.input_proc, "OUTPUT_CLONED", self.main_proc),
            f"time_to_{self.fm_proc}_node": ("ADDED_TO_QUEUE", self.input_proc, "IMAGE_RECEIVED", self.fm_proc),
            # f"{self.fm_proc}_time_cpu2gpu": ("IMAGE_RECEIVED", self.fm_proc, "INPUT_MOVED_TO_GPU", self.fm_proc),
            f"{self.fm_proc}_preprocessing_time": ("IMAGE_RECEIVED", self.fm_proc, "IMAGE_PREPROCESSED", self.fm_proc),
            f"{self.fm_proc}_inference_time": ("IMAGE_PREPROCESSED", self.fm_proc, "INFERENCE_COMPLETED", self.fm_proc),
            # f"{self.fm_proc}_time_gpu2cpu": ("INFERENCE_COMPLETED", self.fm_proc, "OUTPUT_MOVED_TO_CPU", self.fm_proc),
        }

        self.heads_metrics = {}
        for head_proc in self.heads_proc:
            head_metrics = {
                f"{head_proc}_time_to_head": ("INFERENCE_COMPLETED", self.fm_proc, "IMAGE_RECEIVED", head_proc),
                # f"{head_proc}_time_cpu2gpu": ("IMAGE_RECEIVED", head_proc, "INPUT_MOVED_TO_GPU", head_proc),
                f"{head_proc}_inference_time": ("IMAGE_RECEIVED", head_proc, "INFERENCE_COMPLETED", head_proc),
                f"{head_proc}_postprocessing_time": (
                    "INFERENCE_COMPLETED",
                    head_proc,
                    "OUTPUTS_POSTPROCESSED",
                    head_proc,
                ),
                # f"{head_proc}_time_gpu2cpu": ("OUTPUTS_POSTPROCESSED", head_proc, "OUTPUT_MOVED_TO_CPU", head_proc),
                f"{head_proc}_time_to_output": ("OUTPUTS_POSTPROCESSED", head_proc, "OUTPUT_RECEIVED", self.main_proc),
                f"{head_proc}_time_output_cloning": (
                    "OUTPUT_RECEIVED",
                    self.main_proc,
                    "OUTPUT_CLONED",
                    self.main_proc,
                ),
            }
            self.heads_metrics.update(head_metrics)

    @staticmethod
    def time_diff(df, msg1, proc1, msg2, proc2):
        """Calculate the time difference between two log messages."""
        df1 = df[(df["message"] == msg1) & (df["process"] == proc1)]
        df2 = df[(df["message"] == msg2) & (df["process"] == proc2)]

        if len(df1) == 0 or len(df2) == 0:
            return np.nan

        diff = int((df2["timestamp"].values[-1] - df1["timestamp"].values[-1])) * 1e-6  # ns to ms conversion
        if diff > 100:
            print(
                f"Warning: time difference between {proc1}.{msg1} and {proc2}.{msg2} is too high: {int((df2['timestamp'].values[0] - df1['timestamp'].values[0])) * 1e-6} ms"
            )

        return diff

    @staticmethod
    def find_vars(log_msg_id: int, log_msg: str) -> dict:
        syntax = list(MESSAGE)[log_msg_id].value
        test_parts = re.split(r"{\w*}", syntax)
        var_names = re.findall(r"{(\w*)}", syntax)
        vars = []
        log_msg_tmp = log_msg
        p = 0
        for _ in range(len(var_names) + len(test_parts)):
            if p == len(test_parts) - 1 and test_parts[p] == "":
                vars.append(log_msg_tmp)
                p += 1  # not needed, but for clarity, p now equals len(test_parts)
                break
            elif log_msg_tmp.startswith(test_parts[p]):
                log_msg_tmp = log_msg_tmp[len(test_parts[p]) :]
                p += 1
            else:
                # consume the variable
                var = log_msg_tmp[: log_msg_tmp.index(test_parts[p])]
                log_msg_tmp = log_msg_tmp[len(var) :]
                vars.append(var)

        assert len(vars) == len(var_names), f"Expected {len(var_names)} variables, got {len(vars)}"

        kwargs = dict(zip(var_names, vars))
        formatted = list(MESSAGE)[log_msg_id].format(**kwargs)
        original = f"|{log_msg_id}| {log_msg}"
        assert formatted == original, (
            f"Failed to extract variables original log is '{original}', but with extracted variables it is '{formatted}'"
        )

        return dict(zip(var_names, vars))

    def plot_results(self, results):
        df = pd.DataFrame(results)
        fig = px.bar(df, x=df.index, y="total_time", error_y="total_time")
        fig.show()

    def analyze_log(self):
        df = self.parse_log()

        df = df.dropna()
        df = df[df["image_id"] >= 0]  # we treat all values below 0 as invalid
        available_ids = df["image_id"].unique().tolist()

        results = {metric: [] for metric in self.metrics.keys()}
        results.update({metric: [] for metric in self.heads_metrics.keys()})

        for id in sorted(available_ids):
            id_df = df[df["image_id"] == id]

            for metric, (msg1, proc1, msg2, proc2) in self.metrics.items():
                results[metric].append(self.time_diff(id_df, msg1, proc1, msg2, proc2))

            for metric, (msg1, proc1, msg2, proc2) in self.heads_metrics.items():
                results[metric].append(self.time_diff(id_df, msg1, proc1, msg2, proc2))

        means = {metric: np.nanmean(values) for metric, values in results.items()}
        stds = {metric: np.nanstd(values) for metric, values in results.items()}

        for metric, value in means.items():
            print(f"{metric + ':':<45}{value:.1f} ± {stds[metric]:.1f} ms")

        if self.output_file:
            res_df = pd.DataFrame(results)
            res_df.drop(
                columns=["total_time"], inplace=True
            )  # total_time is not needed in the plot (it is a sum of all components)
            plot = res_df.plot(
                title="Time taken for different parts of the pipeline",
                labels={"index": "Image number", "value": "Time (ms)"},
            )
            plot.write_image(self.output_file)

    def parse_log(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.Series([], dtype="datetime64[ms]"),
                "level": pd.Series(
                    [], dtype=pd.CategoricalDtype(categories=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
                ),
                "process": pd.Series([], dtype="str"),
                "message": pd.Series([], dtype=pd.CategoricalDtype(categories=list(MESSAGE.__members__))),
                "image_id": pd.Series([], dtype="int"),
            }
        )
        with open(self.log_file, "r") as f:
            log = f.readlines()

        # find latest engine build successfully message drop all rows before it
        engine_build_idxs = [idx for idx, line in enumerate(log) if MESSAGE.ENGINE_BUILD_SUCCESS.value in line]

        if len(engine_build_idxs) == 0:
            print(
                "WARNING: No ENGINE_BUILD_SUCCESS message found. This is suspicious and likely result in wrong statistics."
            )
            start_idx = 0
        else:
            start_idx = engine_build_idxs[-1]

        for idx, line in enumerate(log[start_idx:]):
            match = re.search(r"\[(.*?)\] \|(\d+)\| ([\S ]*)", line)
            if not match:
                if self.verbose:
                    print(f"Skipping line {idx} {repr(line)}. Unexpected log entry.")
                continue
            log_metadata = match.group(1)
            timestamp, log_level, process_name = log_metadata.split(" - ")
            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")
            log_msg_id = int(match.group(2))
            log_msg = match.group(3)

            vars = self.find_vars(log_msg_id, log_msg)
            df.loc[len(df)] = {
                "timestamp": timestamp,
                "level": log_level,
                "process": process_name,
                "message": list(MESSAGE)[log_msg_id].name,
                "image_id": int(vars.get("n", -1)),
            }

        return df


if __name__ == "__main__":
    logging_config = {"log_file": "{ROS_WORKSPACE}/nn_engine/logs/test.log".format(**os.environ)}
    log_analyzer = LogAnalyzer(logging_config, "results.png", verbose=True)
    log_analyzer.analyze_log()
