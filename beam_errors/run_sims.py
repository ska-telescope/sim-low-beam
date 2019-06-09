#!/usr/bin/env python3

"""Script to run LOW station beam error simulations."""

from __future__ import print_function
import logging
import os
import shutil
import sys
import tempfile

import json
import matplotlib
matplotlib.use('Agg')
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy
import oskar


LOG = logging.getLogger()


def make_vis_data(settings_dict, single_source_offset=-1):
    """Run simulation using supplied settings."""
    settings = oskar.SettingsTree('oskar_sim_interferometer')
    settings.from_dict(settings_dict)
    out = settings['interferometer/ms_filename']
    if os.path.exists(out):
        LOG.info("Skipping simulation, as '%s' already exists", out)
        return

    # Generate single-source sky model if in single-source mode.
    temp_sky = None
    if single_source_offset >= 0:
        ra = float(settings['observation/phase_centre_ra_deg'])
        dec = float(settings['observation/phase_centre_dec_deg'])
        (_, temp_sky) = tempfile.mkstemp(suffix='.osm')
        with open(temp_sky, 'w') as fhan:
            fhan.write("%.5f %.5f 1\n" % (ra, dec - single_source_offset))
        settings['simulator/use_gpus'] = 'false'
        settings['simulator/max_sources_per_chunk'] = '2'
        settings['sky/oskar_sky_model/file'] = temp_sky

    LOG.info("Simulating '%s'", out)
    #print(json.dumps(settings.to_dict(), indent=4))
    sim = oskar.Interferometer(settings=settings)
    sim.run()
    if temp_sky:
        os.remove(temp_sky)


def make_diff_image_stats(ms_name1, ms_name2, image_root=None):
    """Make an image of the difference between two Measurement Sets.

    This function assumes that both Measurement Sets are identical, apart
    from the contents of the visibility DATA column.
    (It will fail horribly otherwise!)
    """
    # Open both Measurement Sets.
    ms1 = oskar.MeasurementSet.open(ms_name1)
    ms2 = oskar.MeasurementSet.open(ms_name2)
    num_rows = ms1.num_rows
    num_channels = ms1.num_channels
    num_stations = ms1.num_stations
    if ms2.num_rows != num_rows or \
            ms2.num_channels != num_channels or \
            ms2.num_stations != num_stations:
        LOG.error("'%s' and '%s' do not have the same dimensions! Aborting.",
                  ms_name1, ms_name2)
        return {}
    num_baselines = num_stations * (num_stations - 1) // 2
    num_chunks = (num_rows + num_baselines - 1) // num_baselines

    # Set up an imager.
    imager = oskar.Imager(precision='double')
    imager.set(fov_deg=5.0, image_size=5000, fft_on_gpu=True)
    imager.set_vis_frequency(ms1.freq_start_hz, ms1.freq_inc_hz, num_channels)
    if image_root is not None:
        imager.output_root = image_root

    LOG.info("Imaging differences between '%s' and '%s'", ms_name1, ms_name2)
    for i_chunk in range(num_chunks):
        start_row = i_chunk * num_baselines
        (uu, vv, ww) = ms1.read_coords(start_row, num_baselines)
        vis1 = ms1.read_vis(start_row, 0, num_channels, num_baselines)
        vis2 = ms2.read_vis(start_row, 0, num_channels, num_baselines)
        vis_out = vis1 - vis2
        imager.update(uu, vv, ww, amps=vis_out,
                      start_channel=0, end_channel=num_channels-1)
    output = imager.finalise(return_images=1)
    del ms1  # Delete the objects (not the Measurement Sets).
    del ms2
    image = output['images'][0]

    LOG.info("Generating image statistics")
    image_size = imager.image_size
    box_size = int(0.1 * image_size)
    centre = image[
        (image_size - box_size)//2:(image_size + box_size)//2,
        (image_size - box_size)//2:(image_size + box_size)//2]
    return {
        'image_max': numpy.max(image),
        'image_min': numpy.min(image),
        'image_maxabs': numpy.max(numpy.abs(image)),
        'image_medianabs': numpy.median(numpy.abs(image)),
        'image_mean': numpy.mean(image),
        'image_rms': numpy.sqrt(numpy.mean(image**2)),
        'image_centre_maxabs': numpy.max(numpy.abs(centre)),
        'image_centre_mean': numpy.mean(centre),
        'image_centre_std': numpy.std(centre),
        'image_centre_rms': numpy.sqrt(numpy.mean(centre**2))
    }


def make_plot(prefix, field_name, metric_key, results,
              axis_gain, axis_phase, specials, single_source_offset):
    """Plot selected results."""
    # Get data for contour plot.
    X, Y = numpy.meshgrid(axis_gain, axis_phase)
    Z = numpy.zeros(X.shape)
    for gain, phase, z in numpy.nditer([X, Y, Z], op_flags=['readwrite']):
        key = '%s_%s_%.3f_dB_%.2f_deg' % (prefix, field_name, gain, phase)
        if key in results:
            z[...] = numpy.log10(results[key][metric_key])

    # Get data for scatter overlay.
    special_gain = []
    special_phase = []
    special_data = []
    special_labels = []
    for label, params in specials.items():
        gain = params['gain_std_dB']
        phase = params['phase_std_deg']
        key = '%s_%s_%.3f_dB_%.2f_deg' % (prefix, field_name, gain, phase)
        if key in results:
            special_gain.append(gain)
            special_phase.append(phase)
            special_data.append(numpy.log10(results[key][metric_key]))
            special_labels.append(label)

    # Contour plot.
    ax1 = plt.subplot(111)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    mi = numpy.min((Z.min(), numpy.array(special_data).min()))
    ma = numpy.max((Z.max(), numpy.array(special_data).max()))
    norm = matplotlib.colors.Normalize(vmin=mi, vmax=ma)
    cp = ax1.contour(X, Y, Z, cmap='plasma', norm=norm)
    clabels = plt.clabel(cp, inline=False, fontsize=10, fmt='%1.1f')
    for txt in clabels:
        txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=2))
    ax1.autoscale(False)

    # Scatter overlay.
    sp = ax1.scatter(special_gain, special_phase, c=special_data,
                     cmap='plasma', norm=norm, zorder=10)
    for i, val in enumerate(special_data):
        ax1.annotate('%.2f' % val,
                     xy=(special_gain[i], special_phase[i]),
                     xytext=(8, -7), textcoords='offset points', zorder=10)
        ax1.annotate("(%s)" % special_labels[i],
                     xy=(special_gain[i], special_phase[i]),
                     xytext=(8, -19), textcoords='offset points', zorder=10)

    # Title and axis labels.
    metric_name = '[ UNKNOWN ]'
    if metric_key == 'image_centre_rms':
        metric_name = 'Central RMS [Jy/beam]'
    elif metric_key == 'image_maxabs':
        metric_name = 'MAX(ABS(image)) [Jy/beam]'
    elif metric_key == 'image_medianabs':
        metric_name = 'MEDIAN(ABS(image)) [Jy/beam]'
    sky_model = 'GLEAM'
    if single_source_offset >= 0:
        sky_model = 'single source'
    plt.title('%s for %s field (%s)' % (metric_name, field_name, sky_model))
    plt.xlabel('Element gain standard deviation [dB]')
    plt.ylabel('Element phase standard deviation [deg]')
    cbar = plt.colorbar(sp, ax=ax1)
    cbar.set_label('log10(%s)' % metric_name)
    plt.savefig('%s_%s_%s.png' % (prefix, field_name, metric_key))
    plt.close('all')


def run_single(prefix_field, settings, single_source_offset,
               gain_std_dB, phase_std_deg, out0_ms, results):
    """Run a single simulation and generate image statistics for it."""
    out = '%s_%.3f_dB_%.2f_deg' % (prefix_field, gain_std_dB, phase_std_deg)
    if out in results:
        LOG.info("Using cached results for '%s'", out)
        return
    out_ms = out + '.ms'
    gain_std = numpy.power(10.0, gain_std_dB / 10.0) - 1.0
    key_prefix = 'telescope/aperture_array/array_pattern/element/'
    settings[key_prefix + 'gain_error_fixed'] = str(gain_std)
    settings[key_prefix + 'phase_error_fixed_deg'] = str(phase_std_deg)
    settings['interferometer/ms_filename'] = out_ms
    make_vis_data(settings, single_source_offset)
    out_image_root = None
    if out == 'GLEAM_100_MHz_EoR0_0.170_dB_1.20_deg':
        out_image_root = out
    results[out] = make_diff_image_stats(out0_ms, out_ms, out_image_root)
    shutil.rmtree(out_ms)  # Delete output MS to save space.


def run_set(prefix, base_settings, fields, axis_gain, axis_phase, specials,
            single_source_offset):
    """Runs a set of simulations."""
    # Iterate over fields.
    for field_name, field in fields.items():
        # Load result set, if it exists.
        prefix_field = prefix + '_' + field_name
        results = {}
        json_file = prefix_field + '_results.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as input_file:
                results = json.load(input_file)

        # Simulate the 'perfect' case.
        out0_ms = '%s_no_errors.ms' % prefix_field
        settings = base_settings.copy()
        settings.update(field)
        settings['interferometer/ms_filename'] = out0_ms
        make_vis_data(settings, single_source_offset)

        # Simulate the error cases.
        for gain_std_dB in axis_gain:
            for phase_std_deg in axis_phase:
                run_single(prefix_field, settings, single_source_offset,
                           gain_std_dB, phase_std_deg, out0_ms, results)

        # Simulate the 'special' error cases.
        for _, params in specials.items():
            gain_std_dB = params['gain_std_dB']
            phase_std_deg = params['phase_std_deg']
            run_single(prefix_field, settings, single_source_offset,
                       gain_std_dB, phase_std_deg, out0_ms, results)

        # Generate plot for the field.
        if single_source_offset >= 0:
            make_plot(prefix, field_name, 'image_centre_rms', results,
                      axis_gain, axis_phase, specials, single_source_offset)
        else:
            make_plot(prefix, field_name, 'image_maxabs', results,
                      axis_gain, axis_phase, specials, single_source_offset)
            make_plot(prefix, field_name, 'image_medianabs', results,
                      axis_gain, axis_phase, specials, single_source_offset)

        # Save result set.
        with open(json_file, 'w') as output_file:
            json.dump(results, output_file, indent=4)


def main():
    """Main function."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)

    # Define common settings.
    base_settings = {
        'simulator': {
            'double_precision': 'true',
            'max_sources_per_chunk': '20000'
        },
        'observation' : {
            'start_frequency_hz': '100e6',
            'frequency_inc_hz': '100e3',
            'length': '06:00:00',
            'num_time_steps': '36'
        },
        'telescope': {
            'input_directory': 'SKA1-LOW_SKO-0000422_Rev3_38m.tm',
            'pol_mode': 'Scalar',
            'aperture_array/array_pattern/element/gain': '1.0'
        },
        'interferometer': {
            'channel_bandwidth_hz': '100e3',
            'time_average_sec': '1.0',
            'max_time_samples_per_block': '4',
            'ignore_w_components': 'true'
        }
    }

    # Define axes of parameter space.
    fields = {
        'EoR0': {
            'sky/oskar_sky_model/file': 'EoR0_20deg_point_source.osm',
            'observation/phase_centre_ra_deg': '0.0',
            'observation/phase_centre_dec_deg': '-27.0',
            'observation/start_time_utc': '2000-01-01 07:00:00.0'
        },
        'EoR1': {
            'sky/oskar_sky_model/file': 'EoR1_20deg_point_source.osm',
            'observation/phase_centre_ra_deg': '60.0',
            'observation/phase_centre_dec_deg': '-30.0',
            'observation/start_time_utc': '2000-01-01 11:00:00.0'
        },
        'EoR2': {
            'sky/oskar_sky_model/file': 'EoR2_20deg_point_source.osm',
            'observation/phase_centre_ra_deg': '170.0',
            'observation/phase_centre_dec_deg': '-10.0',
            'observation/start_time_utc': '2000-01-01 17:45:00.0'
        }
    }
    axis_gain = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    axis_phase = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    specials = {
        'L3 req': {
            'gain_std_dB': 0.17,
            'phase_std_deg': 1.2
        },
        'October data': {
            'gain_std_dB': 0.005,
            'phase_std_deg': 0.17
        },
        'April data': {
            'gain_std_dB': 0.103,
            'phase_std_deg': 0.11
        }
    }

    # Single source simulations.
    single_source_offset = 2.25
    run_set('src1_100_MHz_%.2f_deg' % single_source_offset, base_settings,
            fields, axis_gain, axis_phase, specials, single_source_offset)

    # GLEAM sky model simulations.
    run_set('GLEAM_100_MHz', base_settings,
            fields, axis_gain, axis_phase, specials, -1)


if __name__ == '__main__':
    main()
