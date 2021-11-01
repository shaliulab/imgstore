from typing import Iterable

import tqdm
import pandas as pd
import cv2

class ImgStoreExport:

    def to_videofile(self, output, framerate=None, frame_number: Iterable = None, frame_time: Iterable = None):

        if frame_number is None and frame_time is None:
            raise Exception("Please pass an interval of frame indices or timestamps")
        
        elif not frame_number is None and not frame_time is None:
            raise Exception("Please pass an interval of frame indices or timestamps but only one of them")

        elif frame_number is None and not frame_time is None:
            metadata_df = []

            for chunk in self.chunks:
                metadata = pd.DataFrame(self._index.get_chunk_metadata(chunk))
                metadata["chunk"] = chunk
                metadata_df.append(metadata)

            metadata_df = pd.concat(metadata_df)

            frame_number = []
            for t in frame_time:
                frame_number.append(int(metadata_df.loc[metadata_df["frame_time"] == t]["frame_number"].values[0]))
            
        
        # for readability
        else:
            pass
            # frame_number = frame_number

        frame_number = list(frame_number)
        frame_number[-1] += 1
        frame_number = tuple(frame_number)

        frame_indices = list(range(*frame_number))
        
        if framerate is None:
            framerate = self._fps

        resolution = self._metadata["imgshape"]

        if len(resolution) == 3:
            isColor = True
        else:
            isColor = False
        
        video_writer = cv2.VideoWriter(
            output,
            cv2.VideoWriter_fourcc(*"DIVX"),
            framerate,
            resolution,
            isColor=isColor
        )


        for i, idx in tqmd.tqdm(enumerate(frame_indices)):
            img, (retrieved_idx, _) = self.get_image(idx)
            assert idx == retrieved_idx
            video_writer.write(img)


        video_writer.release()
    