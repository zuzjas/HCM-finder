import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import pandas as pd
import wfdb
from matplotlib import pyplot as plt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinterdnd2 import TkinterDnD, DND_FILES
import seaborn as sns


class Application(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Patients data")
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill="both", expand="true")
        self.geometry("900x500")
        self.search_page = SearchPage(parent=self.main_frame)


# Okno obslugi tabeli
class DataTable(ttk.Treeview):
    def __init__(self, parent):
        super().__init__(parent)
        scroll_Y = tk.Scrollbar(self, orient="vertical", command=self.yview)
        scroll_X = tk.Scrollbar(self, orient="horizontal", command=self.xview)
        self.configure(yscrollcommand=scroll_Y.set, xscrollcommand=scroll_X.set)
        scroll_Y.pack(side="right", fill="y")
        scroll_X.pack(side="bottom", fill="x")
        self.bind("<Double-1>", self.on_double_click)
        self.stored_dataframe = pd.DataFrame(dtype=str)
        self.ecg_plot = plt.plot

    def on_double_click(self, event):
        item = self.selection()[0]
        print("you clicked on", self.item(item)['values'][26])
        self.new_window = ECGWindow().draw_plot(self.plot_ecg_sample(self.item(item)['values'][26]))

    def plot_ecg_sample(path_to_sample):
        path = 'C:/Users/user\/Documents/Python/HCM_finder' \
               '/input/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'\
               + path_to_sample
        sample = wfdb.rdrecord(path, physical=False)
        signal_data = sample.d_signal

        fig, axes = plt.subplots(signal_data.shape[1], 1, figsize=(20, 10))
        for i in range(signal_data.shape[1]):
            sns.lineplot(x=np.arange(signal_data.shape[0]), y=signal_data[:, i], ax=axes[i])

        return fig

    def set_datatable(self, dataframe):
        self.stored_dataframe = dataframe
        self._draw_table(dataframe)

    def _draw_table(self, dataframe):
        self.delete(*self.get_children())
        columns = list(dataframe.columns)
        self.__setitem__("column", columns)
        self.__setitem__("show", "headings")

        for col in columns:
            self.heading(col, text=col)

        df_rows = dataframe.to_numpy().tolist()
        for row in df_rows:
            self.insert("", "end", values=row)
        return None

    def find_value(self, pairs):
        new_df = self.stored_dataframe
        for col, value in pairs.items():
            query_string = f"{col}.str.contains('{value}')"
            new_df = new_df.query(query_string, engine="python")
        self._draw_table(new_df)

    def reset_table(self):
        self._draw_table(self.stored_dataframe)


# Okno wyswietlajace EKG pacjenta
class ECGWindow(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Patient's ECG")
        self.ecg_frame = tk.Frame(self)
        self.ecg_frame.pack(fill="both", expand="true")
        self.geometry("1500x700")

    def draw_plot(self, fig):
        canvas = FigureCanvasTkAgg(fig, self)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas.get_tk_widget().pack()
        canvas.draw()
        canvas.get_tk_widget().pack()


# Okno wyszukiwania
class SearchPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.file_names_listbox = tk.Listbox(parent, selectmode=tk.SINGLE, background="darkgray")
        self.file_names_listbox.place(relheight=1, relwidth=0.25)
        self.file_names_listbox.drop_target_register(DND_FILES)
        self.file_names_listbox.dnd_bind("<<Drop>>", self.drop_inside_list_box)
        self.file_names_listbox.bind("<Double-1>", self._display_file)

        self.search_entrybox = tk.Entry(parent)
        self.search_entrybox.place(relx=0.25, relwidth=0.75)
        self.search_entrybox.bind("<Return>", self.search_table)

        self.data_table = DataTable(parent)
        self.data_table.place(rely=0.05, relx=0.25, relwidth=0.75, relheight=0.95)

        self.path_map = {}

    def drop_inside_list_box(self, event):
        file_paths = self._parse_drop_files(event.data)
        current_listbox_items = set(self.file_names_listbox.get(0, "end"))
        for file_path in file_paths:
            if file_path.endswith(".csv"):
                path_object = Path(file_path)
                file_name = path_object.name
                if file_name not in current_listbox_items:
                    self.file_names_listbox.insert("end", file_name)
                    self.path_map[file_name] = file_path

    def _display_file(self, event):
        file_name = self.file_names_listbox.get(self.file_names_listbox.curselection())
        path = self.path_map[file_name]
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        self.data_table.set_datatable(dataframe=df)

    def _parse_drop_files(self, filename):
        size = len(filename)
        res = []
        name = ""
        idx = 0
        while idx < size:
            if filename[idx] == "{":
                j = idx + 1
                while filename[j] != "}":
                    name += filename[j]
                    j += 1
                res.append(name)
                name = ""
                idx = j
            elif filename[idx] == " " and name != "":
                res.append(name)
                name = ""
            elif filename[idx] != " ":
                name += filename[idx]
            idx += 1
        if name != "":
            res.append(name)
        return res

    def search_table(self, event):
        entry = self.search_entrybox.get()
        if entry == "":
            self.data_table.reset_table()
        else:
            entry_split = entry.split(",")
            column_value_pairs = {}
            for pair in entry_split:
                pair_split = pair.split("=")
                if len(pair_split) == 2:
                    col = pair_split[0]
                    lookup_value = pair_split[1]
                    column_value_pairs[col] = lookup_value
            self.data_table.find_value(pairs=column_value_pairs)


if __name__ == "__main__":
    root = Application()
    root.mainloop()
