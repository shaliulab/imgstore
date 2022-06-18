class AdjustFrameNumberMixin:
    
    def obtain_real_frame_number(self, frame_number):
        if len(self._frame_metadata) == 0:
            self._frame_metadata=self._index.get_all_metadata()
        real_frame_number = self._frame_metadata['frame_number'].index(frame_number)
        return real_frame_number
