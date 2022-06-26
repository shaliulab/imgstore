import numpy as np
import pandas as pd
import tqdm

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

    diff=np.diff(nan_index)
    last_nan = np.where(diff < -1)[0]
    if len(last_nan) == 0:
        last_nan = [nan_index[-1]]
        first_nan = [nan_index[0]]
    else:
        first_nan = []
        for i in range(len(last_nan)):
            if i == 0:
                start = 0
            else:
                start = last_nan[i-1]
            first_nan.append(diff[start:end].index(-1))


    assert len(last_nan) == len(first_nan)    
    for i in range(len(last_nan)):
        data[first_nan[i]:last_nan[i]+1]=column[last_nan[i]+1]

    return data

class MultiStoreCrossIndexMixIn:



    def annotate_chunk(store, store_name, crossindex):
        """
        Annotate chunk for each frame in the crossindex, for the desired store    
        """

        start_of_chunk = getattr(store, f"_{store_name}")._index.get_start_of_chunks()
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


    def build_crossindex(store):
        store.select_store("lowres/metadata.yaml")
        selected_metadata = pd.DataFrame.from_dict(
            store._selected.frame_metadata
        )
        master_metadata = pd.DataFrame.from_dict(
            store._master.frame_metadata
        )
        selected_metadata.columns = ["selected_fn", "selected_ft"]
        master_metadata.columns = ["master_fn", "master_ft"]

        selected_metadata=selected_metadata.loc[selected_metadata["selected_ft"] >= master_metadata["master_ft"][0]]

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
        crossindex=crossindex[~np.isnan(crossindex["master_fn"])]
        crossindex.drop("dummy", axis=1, inplace=True)


        multindex=pd.MultiIndex.from_tuples([
            ("selected", "frame_number"),
            ("selected", "frame_time"),
            ("master", "frame_number"),
            ("master", "frame_time"),
        ], names=["store", "feature"])
        crossindex.columns=multindex

        crossindex=annotate_chunk(store, "master", crossindex)
        crossindex=annotate_chunk(store, "selected", crossindex)

        crossindex[("master", "update")] = [False] + (np.diff(crossindex[("master", "frame_time")]) > 0).tolist()
        crossindex.loc[0, ("master", "update")] = True
        crossindex[("selected", "update")] = True


        return crossindex