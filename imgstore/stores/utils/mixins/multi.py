import logging
import warnings
import os.path
import sqlite3
import numpy as np
import pandas as pd
import tqdm

logger = logging.getLogger(__name__)

def last_value(column):
    """
    Fill np.nans using the last non-nan value seen in the column
    before the occurrence of each nan
    """

    data = np.array([e for e in column])
    non_nan_index=np.where(~np.isnan(column))[0]
    for i in tqdm.tqdm(range(len(non_nan_index))):
        if (i+1) >= len(non_nan_index):
            last_pos = len(data)
        else:
            last_pos = non_nan_index[i+1]

        data[non_nan_index[i]:last_pos] = column[non_nan_index[i]]
    return data

def first_value(column):
    """
    Fill np.nans using the first non-nan value seen in the column
    after the occurrence of each nan
    """
    data = np.array([e for e in column])
    nan_index=np.where(np.isnan(column))[0]

    if len(nan_index) == 0:
        return data

    # print(f"Nan index {nan_index}")

    diff=np.diff(nan_index)
    # print(f"Diff {diff}")
    last_nan = nan_index[np.where(diff > 1)[0]].tolist()

    if len(last_nan) == 0:
        last_nan = [nan_index[-1]]
        first_nan = [nan_index[0]]
    else:
        idx=np.bitwise_and(
            diff == 1,
            [False] + (diff[:-1]!=1).tolist()
        ).tolist() + [True]
        # print(idx)
        if nan_index[0] == 0:
            idx[0]=True

        first_nan = nan_index[idx]

    # print(f"First nan {first_nan}")
    # print(f"Last nan {last_nan}")

    if (len(last_nan) + 1) == len(first_nan):
        last_nan.append(first_nan[-1])  
    for i in range(len(last_nan)):
        data[first_nan[i]:last_nan[i]+1]=column[last_nan[i]+1]

    return data

class MultiStoreCrossIndexMixIn:


    def _make_crossindex(self):

        CROSSINDEX_FILE = os.path.join(self._basedir, "crossindex.db")
        if os.path.exists(CROSSINDEX_FILE):
            logger.info(f"Loading cached crossindex --> {CROSSINDEX_FILE}")
        else:
            logger.info("Generating crossindex. This may take a few seconds")
            self._crossindex=self.build_crossindex()
            logger.info(f"Saving to sqlite file -> {CROSSINDEX_FILE}. This may take a few seconds")
            self._save_to_sqlite(CROSSINDEX_FILE)

        logger.info("Done")

    def _save_to_sqlite(self, CROSSINDEX_FILE):
        with sqlite3.connect(CROSSINDEX_FILE) as conn:
            self._crossindex["master"].to_sql(name="master", con=conn, index_label="id")
            self._crossindex["selected"].to_sql(name="selected", con=conn, index_label="id")
        
        return None

    @property
    def crossindex(self):

        CROSSINDEX_FILE = os.path.join(self._basedir, "crossindex.db")
        if os.path.exists(CROSSINDEX_FILE):
            return self
        elif getattr(self, "_crossindex", None) is None:
            self._make_crossindex()
        
        return self


    @property
    def _conn(self):
        CROSSINDEX_FILE = os.path.join(self._basedir, "crossindex.db")
        if not os.path.exists(CROSSINDEX_FILE):
            self._make_crossindex()

        self.__conn = sqlite3.connect(CROSSINDEX_FILE)
        return self.__conn


    def find_master_fn(self, selected_fn):
        """
        Given a frame number from the selected (higher FPS) store,
        return the frame number that should be retrieved in the master store
        This is conveniently stored in the crossindex
        """
        cur=self._conn.cursor()
        cur.execute(f"SELECT frame_number FROM master WHERE id = {selected_fn};")
        return cur.fetchone()[0]

    def find_selected_fn(self, master_fn):
        """
        Given a frame number from the master (higher spatial resolution) store,
        return the frame number that should be retrieved in the selected store
        This is conveniently stored in the crossindex
        """
        cur=self._conn.cursor()
        cur.execute(f"SELECT id FROM master WHERE frame_number = {master_fn};")
        return cur.fetchone()[0]


    def get_all_master_fn(self):
        cur=self._conn.cursor()
        cur.execute(f"SELECT frame_number FROM master;")
        return cur.fetchall()


    def get_number_of_frames(self, store_name):
        cur=self._conn.cursor()
        cur.execute(f"SELECT frame_number FROM {store_name} ORDER BY id DESC LIMIT 1;")
        return cur.fetchone()[0]

    def annotate_chunk(self, store_name, crossindex):
        """
        Annotate chunk for each frame in the crossindex, for the desired store

        Arguments:

            * store_name (str): Either master or selected
            * crossindex (pd.DataFrame): Table storing for each frame in a store:
            
              1. frame number
              2. frame time

            The columns should have a multiindex where the first level is the store name
            and the second level the column name (frame_number and optionally frame_time)

            This function adds a new column called chunk that annotates
            which chunk does each frame belong to
        
        Returns:

            * crossindex (pd.DataFrame): The same table that was introduced in the input,
            whith a new column "chunk" under the passed store.
        """

        start_of_chunk = getattr(self, f"_{store_name}")._index.get_start_of_chunks()
        multindex=pd.MultiIndex.from_tuples([
            (store_name, "chunk"),
            (store_name, "frame_number"),
        ], names=["store", "feature"])
        crossindex[(store_name, "frame_number")]=crossindex[(store_name, "frame_number")].astype(np.uint64)
        chunks=pd.DataFrame(start_of_chunk, columns=multindex)
        crossindex=crossindex.merge(chunks,  on=[(store_name, "frame_number")], how="left")

        forward=last_value(crossindex[(store_name, "chunk")])#.astype(np.int64)
        backward=first_value(forward).astype(np.int64)

        crossindex[(store_name, "chunk")]=backward
        return crossindex


    def build_crossindex(self):

        """
        Given a multi store with a selected and master stores,
        this function creates a crossindex, a table with the folling structure

        store    master   master         master       master    selected       selected      selected

        feature  update   frame_number   frame_time   chunk     frame_number   frame_time    chunk

        If a frame in the master is labeled as update=True,
        this means any program playing back both from the selected and the master
        needs to update the master with this frame when the corresponding selected frame
        (in the same row) is played
        """


        selected_metadata = pd.DataFrame.from_dict(
            self._selected.frame_metadata
        )
        master_metadata = pd.DataFrame.from_dict(
            self._master.frame_metadata
        )
        selected_metadata.columns = ["selected_fn", "selected_ft"]
        master_metadata.columns = ["master_fn", "master_ft"]

        # selected_metadata=selected_metadata.loc[selected_metadata["selected_ft"] >= master_metadata["master_ft"][0]]

        master_metadata["dummy"]="X"
        selected_metadata["dummy"]="X"

        crossindex = pd.merge_asof(
            selected_metadata,
            master_metadata,
            direction="backward",
            tolerance=1000,
            left_on="selected_ft",
            right_on="master_ft",
            by="dummy"
        )

        # drop frames from the selected store that happen before any in the master
        # since they are useless
        # crossindex=crossindex[~np.isnan(crossindex["master_fn"])]
        crossindex.drop("dummy", axis=1, inplace=True)

        multindex=pd.MultiIndex.from_tuples([
            ("selected", "frame_number"),
            ("selected", "frame_time"),
            ("master", "frame_number"),
            ("master", "frame_time"),
        ], names=["store", "feature"])
        crossindex.columns=multindex

        for store_name in ["master", "selected"]:
            for feat in ["frame_number", "frame_time"]:
                last_nan = np.where(np.isnan(crossindex[(store_name, feat)]))[0].tolist()
                if last_nan:
                    last_nan = last_nan[-1]
                else:
                    continue
                crossindex.loc[:last_nan, (store_name, feat)] = first_value(
                    crossindex.loc[:last_nan, (store_name, feat)].tolist()
                )
                crossindex.loc[:last_nan, (store_name, feat)] = last_value(
                    crossindex.loc[:last_nan, (store_name, feat)].tolist()
                )
                

        crossindex=self.annotate_chunk("master", crossindex)
        crossindex=self.annotate_chunk("selected", crossindex)

        crossindex[("master", "update")] = [False] + (np.diff(crossindex[("master", "frame_time")]) > 0).tolist()
        crossindex.loc[0, ("master", "update")] = True
        crossindex[("selected", "update")] = True

        return crossindex