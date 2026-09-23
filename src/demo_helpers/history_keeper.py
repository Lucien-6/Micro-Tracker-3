#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import contextlib
import os
import json
import time


# ---------------------------------------------------------------------------------------------------------------------
# %% Classes


class HistoryKeeper:
    """
    Class used to manage loading/saving to a 'history' dictionary file
    The point of this being to store & re-use important settings
    (for example, re-using image or model selection paths)
    """

    def __init__(self, file_dunder=None, history_file_name=".history"):

        # Set up history save file pathing
        folder_path = None
        if file_dunder is not None:
            folder_path = os.path.dirname(file_dunder) if os.path.isfile(file_dunder) else file_dunder
        filename = str(os.path.splitext(history_file_name)[0]).lower()
        filepath = filename if folder_path is None else os.path.join(folder_path, filename)

        self._filepath = filepath
        self._history_dict = {}
        self.reload()

    def reload(self):
        """Load and store results from an existing history file"""

        try:
            with open(self._filepath, "r", encoding="utf-8") as infile:
                history_dict = json.load(infile)
            if not isinstance(history_dict, dict):
                history_dict = {}
        except FileNotFoundError:
            history_dict = {}
        except (OSError, ValueError):
            backup = f"{self._filepath}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
            with contextlib.suppress(OSError):
                os.replace(self._filepath, backup)
            print(
                "",
                f"Warning: history file was unreadable and has been reset (backup: {backup})",
                sep="\n",
                flush=True,
            )
            history_dict = {}
        self._history_dict = history_dict

        return self

    def read(self, key):
        """Read from the current copy of history data"""
        have_key = key in self._history_dict.keys()
        loaded_key = self._history_dict.get(key, None)
        return have_key, loaded_key

    def store(self, **key_value_kwargs):
        """Update and save history data"""

        # Check if we can store the new value (validate the dict we actually intend to write!)
        new_history_dict = {**self._history_dict, **key_value_kwargs}
        try:
            encoded = json.dumps(new_history_dict, indent=2)
        except TypeError:
            print("", "ERROR - Cannot store history, invalid as json:", new_history_dict, sep="\n")
            return self

        # Replace the file atomically so a crash mid-write cannot leave a truncated history.
        tmp_path = self._filepath + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as outfile:
                outfile.write(encoded)
            os.replace(tmp_path, self._filepath)
            self._history_dict = new_history_dict
        except OSError as err:
            print("", f"Warning: could not save history ({err})", sep="\n", flush=True)
            with contextlib.suppress(OSError):
                os.remove(tmp_path)

        return self
