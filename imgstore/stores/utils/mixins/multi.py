"""
Provide cross-indexing functionality in a set of synchronized multistores

Problem: play back of a set of (2) multistores that are synchronized
but have a different framerate is not trivial.
This module provides the functionality to generate a cross-index,
a table that shows for each frame on each store, which frame in the other stores
should be mapped to
The index is built in SQLite, and the frame number is indexed,
which helps speeding up lookups several orders of magnitude
"""

import logging
import warnings
import os.path
import sqlite3
import numpy as np
import pandas as pd
import tqdm
import time
import codetiming

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def last_value(column):
    """
    Fill np.nans using the last non-nan value seen in the column
    before the occurrence of each nan
    """

    data = np.array([e for e in column])
    non_nan_index=np.where(~np.isnan(column))[0]
    for i in tqdm.tqdm(range(len(non_nan_index)), desc="Propagating next non-nan value towards the past"):
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
    assert not np.isnan(data).all()
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
            with codetiming.Timer(text="Generated cross-index in {:.8f} seconds", logger=logger.info):
                self._crossindex=self.build_crossindex()
            logger.info(f"Saving to sqlite file -> {CROSSINDEX_FILE}. This may take a few seconds")
            self._save_to_sqlite(CROSSINDEX_FILE)

        logger.info("Done")

    def _save_to_sqlite(self, CROSSINDEX_FILE):
        with sqlite3.connect(CROSSINDEX_FILE) as conn:
            self._crossindex["master"].to_sql(name="master", con=conn, index_label="id")
            self._crossindex["selected"].to_sql(name="selected", con=conn, index_label="id")
            cur=conn.cursor()
            cur.execute("CREATE INDEX master_frame_number ON master (frame_number);")
            cur.execute("CREATE INDEX selected_frame_number ON selected (frame_number);")
        
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

    def find_fn_given_id(self, store, id):
        cur=self._conn.cursor()
        cur.execute(f"SELECT frame_number FROM {store} WHERE id = {id};")
        return cur.fetchone()[0]
    
    def find_id_given_fn(self, store, fn):
        """
        Takes as input a frame number and a store name
        and returns the id on the crossindex tables master and selected
        """

        cur=self._conn.cursor()
        cur.execute(f"SELECT id FROM {store} WHERE frame_number = {fn};")
        return cur.fetchone()[0]

    def find_master_fn(self, selected_fn):
        """
        Given a frame number from the selected (higher FPS) store,
        return the frame number that should be retrieved in the master store
        This is conveniently stored in the crossindex
        """

        id=self.find_id_given_fn("master", selected_fn)
        cur=self._conn.cursor()
        cur.execute(f"SELECT frame_number FROM master WHERE id = {id};")
        data=cur.fetchone()[0]
        return data

    def find_selected_fn(self, master_fn):
        """
        Given a frame number from the master (higher spatial resolution) store,
        return the frame number that should be retrieved in the selected store
        This is conveniently stored in the crossindex
        """

        id=self.find_id_given_fn("master", master_fn)
        cur=self._conn.cursor()
        cur.execute(f"SELECT frame_number FROM selected WHERE id = {id};")
        return cur.fetchone()[0]


    def get_all_master_fn(self):
        cur=self._conn.cursor()
        cur.execute(f"SELECT frame_number FROM master;")
        return cur.fetchall()


    def get_number_of_frames(self):
        cur=self._conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM master;")
        return cur.fetchone()[0]

    def get_number_of_frames_in_chunk(self, chunk):
        cur=self._conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM master WHERE chunk={chunk};")
        return cur.fetchone()[0]

    def get_starting_frame_of_chunk(self, chunk):
        cur=self._conn.cursor()
        cur.execute(f"SELECT id FROM master WHERE chunk={chunk} LIMIT 1;")
        return cur.fetchone()[0]

    def get_ending_frame_of_chunk(self, chunk):
        cur=self._conn.cursor()
        cur.execute(f"SELECT id FROM master WHERE chunk={chunk};")
        return cur.fetchall()[-1][0]
 
    def get_refresh_ids(self):
        cur=self._conn.cursor()
        cur.execute("SELECT id FROM master WHERE refresh=1;")
        return cur.fetchall()
        

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

        logger.info("Loading metadata from selected store")
        selected_metadata = pd.DataFrame.from_dict(
            self._selected.frame_metadata
        )

        logger.info("Loading metadata from master store")
        master_metadata = pd.DataFrame.from_dict(
            self._master.frame_metadata
        )
        selected_metadata.columns = ["selected_fn", "selected_ft"]
        master_metadata.columns = ["master_fn", "master_ft"]

        # selected_metadata=selected_metadata.loc[selected_metadata["selected_ft"] >= master_metadata["master_ft"][0]]

        master_metadata["dummy"]="X"
        selected_metadata["dummy"]="X"


        logger.info("Performing first merge_asof")
        crossindex = pd.merge_asof(
            selected_metadata,
            master_metadata,
            direction="backward",
            tolerance=1000,
            left_on="selected_ft",
            right_on="master_ft",
            by="dummy"
        )

        logger.info("Performing second merge_asof")
        # perform a second merge
        # so situations where the master camera 
        # managed to produced to frames while the
        # selected camera didn't
        # dont end up discarding the first of the frames from master
        crossindex2=pd.merge_asof(
            master_metadata,
            selected_metadata,
            direction="backward",
            tolerance=1000,
            left_on="master_ft",
            right_on="selected_ft",
            by="dummy"
        )
        logger.info("Done")
        logger.info("Computing frame_number of missing master frames")
        missing_master_fns = set(crossindex2["master_fn"]).difference(crossindex["master_fn"])

        logger.info("Adding missing master frames to cross-index")
        crossindex=pd.concat([
            crossindex,
            crossindex2.loc[crossindex2["master_fn"].isin(missing_master_fns)]
        ])

        logger.info("Sorting cross-index")
        crossindex.sort_values(["selected_ft", "master_ft"], inplace=True)       
        crossindex.reset_index(inplace=True)
        crossindex.drop("index", axis=1, inplace=True)

        assert (np.diff(crossindex["master_fn"][~np.isnan(crossindex["master_fn"])]) < 2).all()

        # drop frames from the selected store that happen before any in the master
        # since they are useless
        # crossindex=crossindex[~np.isnan(crossindex["master_fn"])]
        crossindex.drop("dummy", axis=1, inplace=True)
        logger.info("Done merging")


        multindex=pd.MultiIndex.from_tuples([
            ("selected", "frame_number"),
            ("selected", "frame_time"),
            ("master", "frame_number"),
            ("master", "frame_time"),
        ], names=["store", "feature"])
        crossindex.columns=multindex

        for store_name in ["master", "selected"]:
            for feature in ["frame_number", "frame_time"]:
                logger.info(f"Filling nan values in {store_name}, {feature}")
                nans = np.where(np.isnan(crossindex[(store_name, feature)]))[0].tolist()
                if nans:
                    last_nan = nans[-1]
                else:
                    continue
                while len(nans) != 0:

                    try:
                        crossindex.loc[:last_nan+1, (store_name, feature)] = first_value(
                            crossindex.loc[:last_nan+1, (store_name, feature)].tolist()
                        )
                    except IndexError:
                        pass

                    crossindex.loc[:last_nan, (store_name, feature)] = last_value(
                        crossindex.loc[:last_nan, (store_name, feature)].tolist()
                    )
                    nans = np.where(np.isnan(crossindex[(store_name, feature)]))[0]
                    if nans:
                        nans = nans.tolist()
                    else:
                        break

        logger.info("Annotating chunk in master store")
        crossindex=self.annotate_chunk("master", crossindex)
        logger.info("Annotating chunk in selected store")
        crossindex=self.annotate_chunk("selected", crossindex)

        crossindex[("master", "refresh")] = [False] + (np.diff(crossindex[("master", "frame_time")]) > 0).tolist()
        crossindex.loc[0, ("master", "refresh")] = True
        crossindex[("selected", "refresh")] = True

        return crossindex