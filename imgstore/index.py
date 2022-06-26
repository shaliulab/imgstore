from importlib.resources import path
import os.path
import sqlite3
import logging
import operator
import warnings

import yaml
import numpy as np

from .constants import FRAME_MD, SQLITE3_INDEX_FILE, FRAME_NUMBER_RESET


def _load_index(path_without_extension):
    for extension in ('.npz', '.yaml'):
        path = path_without_extension + extension
        if os.path.exists(path):
            if extension == '.yaml':
                with open(path, 'rt') as f:
                    dat = yaml.safe_load(f)
                    return {k: dat[k] for k in FRAME_MD}
            elif extension == '.npz':
                with open(path, 'rb') as f:
                    dat = np.load(f)
                    return {k: dat[k].tolist() for k in FRAME_MD}

    raise IOError('could not find index %s' % path_without_extension)


# noinspection SqlNoDataSourceInspection,SqlDialectInspection,SqlResolve
class ImgStoreIndex(object):

    VERSION = '1'

    log = logging.getLogger('imgstore.index')

    def __init__(self, db=None):
        self._conn = db
        self.path=None

        cur = self._conn.cursor()
        cur.execute('pragma query_only = ON;')

        cur.execute('SELECT value FROM index_information WHERE name = ?', ('version', ))
        v, = cur.fetchone()
        if v != self.VERSION:
            raise IOError('incorrect index version: %s vs %s' % (v, self.VERSION))

        cur.execute('SELECT COUNT(1) FROM frames')
        self.frame_count, = cur.fetchone()


        if self.frame_count:
            self.frame_time_max = self._summary('frame_time_max')
            self.frame_time_min = self._summary('frame_time_min')
            self.frame_max = self._summary('frame_max')
            self.frame_min = self._summary('frame_min')

            # keep back compat for nan as types (inf -> nan)
            if not np.isreal(self.frame_max):
                self.frame_max = np.nan
            if not np.isreal(self.frame_min):
                self.frame_min = np.nan
        else:
            self.frame_max = self.frame_min = np.nan
            self.frame_time_max = self.frame_time_min = 0.0

        self.log.debug('frame range %f -> %f' % (self.frame_min, self.frame_max))

        # # all chunks in the store [0,1,2, ... ]
        cur.execute('SELECT chunk FROM chunks ORDER BY chunk;')
        self._chunks = tuple(row[0] for row in cur)

    def _summary(self, _what):
        cur = self._conn.cursor()
        cur.execute('SELECT value FROM summary WHERE name = ?', (_what,))
        return cur.fetchone()[0]


    @classmethod
    def read_records(cls, chunk_n_and_chunk_paths):
            records = []
            frame_count = 0
            frame_max = -np.inf
            frame_min = np.inf
            frame_time_max = -np.inf
            frame_time_min = np.inf
            chunks = []

            for chunk_n, chunk_path in sorted(chunk_n_and_chunk_paths, key=operator.itemgetter(0)):
                try:
                    idx = _load_index(chunk_path)
                    if FRAME_NUMBER_RESET:
                        idx["frame_number"] = list(range(frame_count, frame_count+len(idx["frame_number"])))

                except IOError:
                    cls.log.warn('missing index for chunk %s' % chunk_n)
                    continue

                if not idx['frame_number']:
                    # empty chunk
                    continue

                frame_count += len(idx['frame_number'])
                frame_time_min = min(frame_time_min, np.min(idx['frame_time']))
                frame_time_max = max(frame_time_max, np.max(idx['frame_time']))
                frame_min = min(frame_min, np.min(idx['frame_number']))
                frame_max = max(frame_max, np.max(idx['frame_number']))

                try:
                    records += [(chunk_n, i, fn, ft) for i, (fn, ft) in enumerate(zip(idx['frame_number'],
                                                                                        idx['frame_time']))]
                    chunks += [(chunk_n, chunk_path)]
                except TypeError:
                    cls.log.error('corrupt chunk', exc_info=True)
                    continue
            
            return records, chunks, (frame_min, frame_max, frame_time_min, frame_time_max)


    @classmethod
    def populate_database(cls, path, records_and_stats):
        
        records, chunks, stats = records_and_stats
        (frame_min, frame_max, frame_time_min, frame_time_max) = stats

        db = sqlite3.connect(path, check_same_thread=False)

        cls.create_database(db)
        cur = db.cursor()

        cur.executemany('INSERT INTO frames VALUES (?,?,?,?)', records)

        for i in range(len(chunks)):
            cur.execute('INSERT INTO chunks VALUES (?, ?)', chunks[i])
            db.commit()

        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_time_min', float(frame_time_min)))
        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_time_max', float(frame_time_max)))
        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_min', float(frame_min)))
        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_max', float(frame_max)))
        db.commit()
        return db
            
    @classmethod
    def read_index_and_populate_database(cls, chunk_n_and_chunk_paths):
        path = os.path.join(os.path.dirname(chunk_n_and_chunk_paths[0][1]), SQLITE3_INDEX_FILE)
        records_and_stats = cls.read_records(chunk_n_and_chunk_paths)
        cls.populate_database(path, records_and_stats)
        db=cls.populate_database(':memory:', records_and_stats)
        return db, path


    @classmethod
    def create_database(cls, conn):
        c = conn.cursor()
        # Create tables
        c.execute('CREATE TABLE frames '
                  '(chunk INTEGER, frame_idx INTEGER, frame_number INTEGER, frame_time REAL)')
        c.execute('CREATE TABLE chunks '
                  '(chunk INTEGER, chunk_path TEXT)')
        c.execute('CREATE TABLE index_information '
                  '(name TEXT, value TEXT)')
        c.execute('CREATE TABLE summary '
                  '(name TEXT, value REAL)')
        c.execute('INSERT into index_information VALUES (?, ?)', ('version', cls.VERSION))
        c.execute("CREATE INDEX chunk_index ON frames (chunk, frame_idx);")
        conn.commit()

    @classmethod
    def new_from_chunks(cls, chunk_n_and_chunk_paths):
        db, path=cls.read_index_and_populate_database(chunk_n_and_chunk_paths)
        index=cls(db)
        index.path=path
        return index

    @classmethod
    def new_from_chunks_old(cls, chunk_n_and_chunk_paths):
        db = sqlite3.connect(':memory:', check_same_thread=False)
        cls.create_database(db)

        frame_count = 0
        frame_max = -np.inf
        frame_min = np.inf
        frame_time_max = -np.inf
        frame_time_min = np.inf

        cur = db.cursor()

        for chunk_n, chunk_path in sorted(chunk_n_and_chunk_paths, key=operator.itemgetter(0)):
            try:
                idx = _load_index(chunk_path)
            except IOError:
                cls.log.warn('missing index for chunk %s' % chunk_n)
                continue

            if not idx['frame_number']:
                # empty chunk
                continue

            frame_count += len(idx['frame_number'])
            frame_time_min = min(frame_time_min, np.min(idx['frame_time']))
            frame_time_max = max(frame_time_max, np.max(idx['frame_time']))
            frame_min = min(frame_min, np.min(idx['frame_number']))
            frame_max = max(frame_max, np.max(idx['frame_number']))

            try:
                records = [(chunk_n, i, fn, ft) for i, (fn, ft) in enumerate(zip(idx['frame_number'],
                                                                                 idx['frame_time']))]
            except TypeError:
                cls.log.error('corrupt chunk', exc_info=True)
                continue

            cur.executemany('INSERT INTO frames VALUES (?,?,?,?)', records)
            cur.execute('INSERT INTO chunks VALUES (?, ?)', (chunk_n, chunk_path))

            db.commit()

        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_time_min', float(frame_time_min)))
        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_time_max', float(frame_time_max)))
        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_min', float(frame_min)))
        cur.execute('INSERT INTO summary VALUES (?,?)', ('frame_max', float(frame_max)))

        db.commit()

        return cls(db)


    @classmethod
    def new_from_file(cls, path):
        db = sqlite3.connect(path, check_same_thread=False)
        index=cls(db)
        index.path=path
        return index

    @staticmethod
    def _get_metadata(cur):
        md = {'frame_number': [], 'frame_time': []}
        for row in cur:
            md['frame_number'].append(row[0])
            md['frame_time'].append(row[1])
        return md

    @property
    def chunks(self):
        """ the number of non-empty chunks that contain images """
        return self._chunks

    def to_file(self, path):
        db = sqlite3.connect(path)
        with db:
            for line in self._conn.iterdump():
                # let python handle the transactions
                if line not in ('BEGIN;', 'COMMIT;'):
                    db.execute(line)
        db.commit()
        db.close()

    def get_all_metadata(self):
        cur = self._conn.cursor()
        cur.execute("SELECT frame_number, frame_time FROM frames ORDER BY rowid;")
        return self._get_metadata(cur)

    def get_chunk_metadata(self, chunk_n):
        cur = self._conn.cursor()
        cur.execute("SELECT frame_number, frame_time FROM frames WHERE chunk = ? ORDER BY rowid;", (chunk_n, ))
        return self._get_metadata(cur)

    def find_chunk(self, what, value):
        assert what in ('frame_number', 'frame_time', 'index')
        cur = self._conn.cursor()

        if what == 'index':
            cur.execute("SELECT chunk, frame_idx FROM frames ORDER BY rowid LIMIT 1 OFFSET {};".format(int(value)))
        else:
            cur.execute("SELECT chunk, frame_idx FROM frames WHERE {} = ?;".format(what), (value, ))

        try:
            chunk_n, frame_idx = cur.fetchone()
        except TypeError:  # no result
            return -1, -1

        return chunk_n, frame_idx

    def find_chunk_nearest(self, what, value, past=True, future=True):
        # future: frame_time >= value
        # past: frame_time <= value

        assert what in ('frame_number', 'frame_time')
        cur = self._conn.cursor()

        if what=="frame_time":
            cur.execute("SELECT frame_time from frames LIMIT 1;")
            ft=cur.fetchone()[0]
            if type(ft) is float:
                value=float(value)

        if past and future:
            cur.execute("SELECT chunk, frame_idx FROM frames ORDER BY ABS(? - {}) LIMIT 1;".format(what), (value, ))
        elif not future:
            # we want only frame_time <= value
            # frame_time - value will be max 0
            cur.execute("SELECT chunk, frame_idx FROM frames WHERE ({} - ?) <= 0 ORDER BY ABS(? - {}) LIMIT 1;".format(what, what), (value, value))
        elif not past:
            # we want only frame_time >= value
            # frame_time - value will be min 0
            cur.execute("SELECT chunk, frame_idx FROM frames WHERE ({} - ?) >= 0 ORDER BY ABS(? - {}) LIMIT 1;".format(what, what), (value, value))

        chunk_n, frame_idx = cur.fetchone()
        return chunk_n, frame_idx

    def find_all(self, what, value, exact_only=True, past=True, future=True):
        assert what in ('frame_number', 'frame_time')
        cur = self._conn.cursor()
        
        if exact_only:
            filter= f"WHERE {what} = ?"
            val_tuple = (value,)

        elif past and future:
            filter=f" ORDER BY ABS(? - {what}) LIMIT 1"
            val_tuple = (value,)

        elif not future:
            filter=f"WHERE ({what} - ?) <= 0 ORDER BY ABS(? - {what}) LIMIT 1"
            val_tuple = (value, value,)

        elif not past:
            filter=f"WHERE ({what} - ?) >= 0 ORDER BY ABS(? - {what}) LIMIT 1"
            val_tuple = (value, value,)


        cmd=f"SELECT * FROM frames {filter};"
        cur.execute(cmd, val_tuple)

        data = cur.fetchone()
        if data is None:

            if what == "frame_number":
                _what=["frame_min", "frame_max"]
            elif what == "frame_time":
                _what =["frame_time_min", "frame_time_max"]
            error_msg=f"Cannot find {what} set to {value}\n"
            for w in _what:
                error_msg+=f"{w}={str(self._summary(w))} "
            raise IndexError(error_msg)
        return data

    
    def get_start_of_chunks(self):
        """
        Return the chunk and starting frame number
        where a new chunk is created as a list of tuples
        """
        cur = self._conn.cursor()
        cur.execute("SELECT chunk, frame_number from frames where frame_idx == 0;")
        return cur.fetchall()