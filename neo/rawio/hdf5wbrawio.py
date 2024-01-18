"""
ExampleRawIO is a class of a fake example.
This is to be used when coding a new RawIO.


Rules for creating a new class:
  1. Step 1: Create the main class
    * Create a file in **neo/rawio/** that ends with "rawio.py"
    * Create the class that inherits from BaseRawIO
    * copy/paste all methods that need to be implemented.
    * code hard! The main difficulty is `_parse_header()`.
      In short you have to create a mandatory dict that
      contains channel information::

            self.header = {}
            self.header['nb_block'] = 2
            self.header['nb_segment'] = [2, 3]
            self.header['signal_streams'] = signal_streams
            self.header['signal_channels'] = signal_channels
            self.header['spike_channels'] = spike_channels
            self.header['event_channels'] = event_channels

  2. Step 2: RawIO test:
    * create a file in neo/rawio/tests with the same name with "test_" prefix
    * copy paste neo/rawio/tests/test_examplerawio.py and do the same

  3. Step 3: Create the neo.io class with the wrapper
    * Create a file in neo/io/ that ends with "io.py"
    * Create a class that inherits both your RawIO class and the BaseFromRaw class
    * copy/paste from neo/io/exampleio.py

  4.Step 4: IO test
    * create a file in neo/test/iotest with the same name as previously with "test_" prefix
    * copy/paste from neo/test/iotest/test_exampleio.py

"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

import h5py
import numpy as np
import quantities as pq


class Hdf5WbRawIO(BaseRawIO):
    """
    Class for reading data from a misformatted .nwb file as an HDF5 via Raw IO.
    """
    extensions = ['nwb']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        # note that this filename is ued in self._source_name
        self.filename = filename

    def _source_name(self):
        # this function is used by __repr__
        # for general cases self.filename is good
        # But for URL you could mask some part of the URL to keep
        # the main part.
        return self.filename

    def _parse_header(self):
        # This is the central part of a RawIO
        # we need to collect from the original format all
        # information required for fast access
        # at any place in the file
        # In short `_parse_header()` can be slow but
        # `_get_analogsignal_chunk()` needs to be as fast as possible

        try:
            nwb = h5py.File(self.filename, mode=self.nwb_file_mode)
        except ValueError:
            print("Error: Unable to read this version of HDF5/NWB file.")
            print("Please convert to a later HDF5/NWB format.")
            raise

        # identify each probe under genera/devices with a signal stream
        signal_streams = []
        probe_attrs = []
        for stream_id, name in enumerate(nwb['general/devices']):
            signal_streams.append((name, stream_id))
            lfp_data = nwb['acquisition/probe_%d_lfp/probe_%d_lfp_data' % (stream_id, stream_id)]
            lfp_times = lfp_data['timestamps']
            probe_attrs.append({
                "dtype": lfp_data['data'].dtype,
                "sr": (1 / (lfp_times[1:] - lfp_times[:-1])).mean().astype(int),
                "t_start": lfp_times[0], "t_stop": lfp_times[-1],
                "size": len(lfp_times)
            })
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        self._t_starts = [[np.round(probe_attrs[0]["t_start"], 3)]]
        self._t_stops = [[np.round(probe_attrs[0]["t_stop"], 3)]]
        self._signal_attrs = [[probe_attrs]]

        # gain/offset/units are really important because
        # the scaling to real value will be done with that
        # The real signal will be evaluated as `(raw_signal * gain + offset) * pq.Quantity(units)`
        signal_channels = []
        # Signal channels for Local Field Potentials
        electrodes = nwb['general/extracellular_ephys/electrodes']
        for chan_id in electrodes['id'][:]:
            ch_name = '{}/ch{}'.format(electrodes['group_name'].decode(), c)
            # The channel id is just the electrode's global index
            probe_id = electrodes['probe_id'][chan_id]
            sr = probe_attrs[probe_id]['sr']
            dtype = probe_attrs[probe_id]['dtype']
            units = 'uV'
            gain = 1.
            offset = 0.
            # Group channels according to their probes, which share a dtype, sampling rate, and units
            signal_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, probe_id))
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # Fetch the timestamps to get the sampling rate for the spike train
        spike_timestamps = nwb['processing/spike_train/spike_train_data/timestamps']
        spike_sr = (1 / (spike_timestamps[1:] - spike_timestamps[:-1])).mean().astype(int)
        # This is mandatory!!!!
        # Note that if there is no waveform at all in the file
        # then wf_units/wf_gain/wf_offset/wf_left_sweep/wf_sampling_rate
        # can be set to any value because _spike_raw_waveforms
        # will return None
        spike_channels = []
        for c, unit in enumerate(nwb['units/id'][:]):
            unit_name = 'unit{}'.format(unit)
            unit_id = '#{}'.format(unit)
            wf_units = 'uV'
            wf_gain = 1.
            wf_offset = 0.
            wf_left_sweep = 0.
            wf_sampling_rate = spike_sr
            spike_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                   wf_offset, wf_left_sweep, wf_sampling_rate))
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # creating event/epoch channels out of all intervals
        # This is mandatory!!!!
        # In RawIO epoch and event are dealt with in the same way.
        event_channels = []
        for i, intervals in enumerate(nwb['intervals'].keys()):
            event_channels.append((intervals, 'ep_%d' % i, 'epoch'))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # fill information into the header dict
        # This is mandatory!!!!!
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        # insert some annotations/array_annotations at some place
        # at neo.io level. IOs can add annotations
        # to any object. To keep this functionality with the wrapper
        # BaseFromRaw you can add annotations in a nested dict.

        # `_generate_minimal_annotations()` must be called to generate the nested
        # dict of annotations/array_annotations
        self._generate_minimal_annotations()
        # this pprint lines really help with understanding the nested (and sometimes complicated) dict
        # from pprint import pprint
        # pprint(self.raw_annotations)

        # Until here all mandatory operations for setting up a rawio are implemented.
        # The following lines provide additional, recommended annotations for the
        # final neo objects.
        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann['name'] = 'Block #{}'.format(block_index)
        bl_ann['block_extra_info'] = 'This is the block {}'.format(block_index)
        for seg_index in range(self.header['nb_segment'][block_index]):
            seg_ann = bl_ann['segments'][seg_index]
            seg_ann['name'] = 'Seg #{} Block #{}'.format(seg_index, block_index)
            seg_ann['seg_extra_info'] = 'This is the seg {} of block {}'.format(
                seg_index, block_index
            )

        nwb.close()

    def _segment_t_start(self, block_index, seg_index):
        # this must return a float scaled in seconds
        # this t_start will be shared by all objects in the segment
        # except AnalogSignal
        return self._t_starts[block_index][seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        # this must return a float scaled in seconds
        return self._t_stops[block_index][seg_index]

    def _get_signal_size(self, block_index, seg_index, stream_index):

        # You should return the signal size depending on the block_index and
        # segment_index. This must return an int = the number of samples

        # Note that channel_indexes can be ignored for most cases
        # except for the case of several sampling rates.
        return self._signal_attrs[block_index][seg_index][stream_index]["size"]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        # This give the t_start of a signal.
        # Very often this is equal to _segment_t_start but not
        # always.
        # this must return a float scaled in seconds

        # Note that channel_indexes can be ignored for most cases
        # except for the case of several sampling rates.

        # Here this is the same.
        # this is not always the case
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        raise (NotImplementedError)
        # this must return a signal chunk in a signal stream
        # limited with i_start/i_stop (can be None)
        # channel_indexes can be None (=all channel in the stream) or a list or numpy.array
        # This must return a numpy array 2D (even with one channel).
        # This must return the original dtype. No conversion here.
        # This must be as fast as possible.
        # To speed up this call all preparatory calculations should be implemented
        # in _parse_header().

        # Here we are lucky:  our signals are always zeros!!
        # it is not always the case :)
        # internally signals are int16
        # conversion to real units is done with self.header['signal_channels']

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = 100000

        if i_start < 0 or i_stop > 100000:
            # some checks
            raise IndexError("I don't like your jokes")

        if channel_indexes is None:
            nb_chan = 8
        elif isinstance(channel_indexes, slice):
            channel_indexes = np.arange(8, dtype='int')[channel_indexes]
            nb_chan = len(channel_indexes)
        else:
            channel_indexes = np.asarray(channel_indexes)
            if any(channel_indexes < 0):
                raise IndexError('bad boy')
            if any(channel_indexes >= 8):
                raise IndexError('big bad wolf')
            nb_chan = len(channel_indexes)

        raw_signals = np.zeros((i_stop - i_start, nb_chan), dtype='int16')
        return raw_signals

    def _spike_count(self, block_index, seg_index, spike_channel_index):
        # Must return the nb of spikes for given (block_index, seg_index, spike_channel_index)
        # we are lucky:  our units have all the same nb of spikes!!
        # it is not always the case
        raise (NotImplementedError)
        nb_spikes = 20
        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        # In our IO, timestamp are internally coded 'int64' and they
        # represent the index of the signals 10kHz
        # we are lucky: spikes have the same discharge in all segments!!
        # incredible neuron!! This is not always the case

        # the same clip t_start/t_start must be used in _spike_raw_waveforms()

        raise (NotImplementedError)
        ts_start = (self._segment_t_start(block_index, seg_index) * 10000)

        spike_timestamps = np.arange(0, 10000, 500) + ts_start

        if t_start is not None or t_stop is not None:
            # restrict spikes to given limits (in seconds)
            lim0 = int(t_start * 10000)
            lim1 = int(t_stop * 10000)
            mask = (spike_timestamps >= lim0) & (spike_timestamps <= lim1)
            spike_timestamps = spike_timestamps[mask]

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        raise (NotImplementedError)
        # must rescale to seconds, a particular spike_timestamps
        # with a fixed dtype so the user can choose the precision they want.
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= 10000.  # because 10kHz
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, spike_channel_index,
                                 t_start, t_stop):
        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)
        # in the original dtype
        # this must be as fast as possible.
        # the same clip t_start/t_start must be used in _spike_timestamps()

        # If there there is no waveform supported in the
        # IO them _spike_raw_waveforms must return None

        # In our IO waveforms come from all channels
        # they are int16
        # conversion to real units is done with self.header['spike_channels']
        # Here, we have a realistic case: all waveforms are only noise.
        # it is not always the case
        # we get 20 spikes with a sweep of 50 (5ms)

        raise (NotImplementedError)
        # trick to get how many spike in the slice
        ts = self._get_spike_timestamps(block_index, seg_index,
                                        spike_channel_index, t_start, t_stop)
        nb_spike = ts.size

        np.random.seed(2205)  # a magic number (my birthday)
        waveforms = np.random.randint(low=-2**4, high=2**4, size=nb_spike * 50, dtype='int16')
        waveforms = waveforms.reshape(nb_spike, 1, 50)
        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        raise (NotImplementedError)
        # event and spike are very similar
        # we have 2 event channels
        if event_channel_index == 0:
            # event channel
            return 6
        elif event_channel_index == 1:
            # epoch channel
            return 10

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        raise (NotImplementedError)
        # the main difference between spike channel and event channel
        # is that for event channels we have 3D numpy array (timestamp, durations, labels) where
        # durations must be None for 'event'
        # label must a dtype ='U'

        # in our IO events are directly coded in seconds
        seg_t_start = self._segment_t_start(block_index, seg_index)
        if event_channel_index == 0:
            timestamp = np.arange(0, 6, dtype='float64') + seg_t_start
            durations = None
            labels = np.array(['trigger_a', 'trigger_b'] * 3, dtype='U12')
        elif event_channel_index == 1:
            timestamp = np.arange(0, 10, dtype='float64') + .5 + seg_t_start
            durations = np.ones((10), dtype='float64') * .25
            labels = np.array(['zoneX'] * 5 + ['zoneZ'] * 5, dtype='U12')

        if t_start is not None:
            keep = timestamp >= t_start
            timestamp, labels = timestamp[keep], labels[keep]
            if durations is not None:
                durations = durations[keep]

        if t_stop is not None:
            keep = timestamp <= t_stop
            timestamp, labels = timestamp[keep], labels[keep]
            if durations is not None:
                durations = durations[keep]

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        raise (NotImplementedError)
        # must rescale to seconds for a particular event_timestamps
        # with a fixed dtype so the user can choose the precision they want.

        # really easy here because in our case it is already in seconds
        event_times = event_timestamps.astype(dtype)
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        raise (NotImplementedError)
        # really easy here because in our case it is already in seconds
        durations = raw_duration.astype(dtype)
        return durations
