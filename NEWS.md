
0.3.1
================

* Define entrypoint `imgstore-codecs` to check which codecs are available. The output of this command gets saved to the config file `~/.config/imgstore/imgstore.yml`
* Code is refactored to split the `stores.py` module into multiple isolated modules much easier to work with


0.4.1
==========================

* Implement future and past in get_nearest_image

0.4.2
=========================

* Implement FRAME_NUMBER_RESET so the user can decide to reset or "remake" the frame_number and replace what is stored in the index


0.4.6
============================


* Add main_muxer


0.4.17

=============================

Decouple master/selected from main/secondary

* master store is the feed passed as main view. it gets renamed to main later downstream,
and master becomes then highres always
* selected store is the store added as a secondary view. it gets renamed to secondary later downstream,
and selected becomes then highspeed always