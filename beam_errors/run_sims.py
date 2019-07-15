#!/usr/bin/env python3

"""Script to run LOW station beam error simulations."""

from __future__ import print_function, division
import json
import logging
import os
import sys

from astropy.time import Time, TimeDelta
import matplotlib
matplotlib.use('Agg')
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy
import oskar


LOG = logging.getLogger()


def append_bright_sources(sky):
    """Appends bright A-team sources to sky model."""
    # A list of bright sources.
    # Sgr A: guesstimates only!
    # For A: data from the Molonglo Southern 4 Jy sample (VizieR).
    # Others from GLEAM reference paper, Hurley-Walker et al. (2017), Table 2.
    # pylint: disable=bad-whitespace
    sky_bright = oskar.Sky.from_array(numpy.array((
        [266.41683, -29.00781,  2000,0,0,0,   0,    0,    0, 3600, 3600, 0],
        [ 50.67375, -37.20833,   528,0,0,0, 178e6, -0.51, 0, 0, 0, 0],  # For
        [201.36667, -43.01917,  1370,0,0,0, 200e6, -0.50, 0, 0, 0, 0],  # Cen
        [139.52500, -12.09556,   280,0,0,0, 200e6, -0.96, 0, 0, 0, 0],  # Hyd
        [ 79.95833, -45.77889,   390,0,0,0, 200e6, -0.99, 0, 0, 0, 0],  # Pic
        [252.78333,   4.99250,   377,0,0,0, 200e6, -1.07, 0, 0, 0, 0],  # Her
        [187.70417,  12.39111,   861,0,0,0, 200e6, -0.86, 0, 0, 0, 0],  # Vir
        [ 83.63333,  22.01444,  1340,0,0,0, 200e6, -0.22, 0, 0, 0, 0],  # Tau
        [299.86667,  40.73389,  7920,0,0,0, 200e6, -0.78, 0, 0, 0, 0],  # Cyg
        [350.86667,  58.81167, 11900,0,0,0, 200e6, -0.41, 0, 0, 0, 0]   # Cas
        )))
    sky.append(sky_bright)


def get_start_time(ra0_deg, length_sec):
    """Returns optimal start time for field RA and observation length."""
    t = Time('2000-01-01 00:00:00', scale='utc', location=('116.764d', '0d'))
    dt_hours = 24.0 - t.sidereal_time('apparent').hour + (ra0_deg / 15.0)
    start = t + TimeDelta(dt_hours * 3600.0 - length_sec / 2.0, format='sec')
    return start.value


def make_vis_data(settings, sky, tel):
    """Run simulation using supplied settings."""
    if os.path.exists(settings['interferometer/oskar_vis_filename']):
        LOG.info("Skipping simulation, as output data already exist.")
        return
    LOG.info("Simulating %s", settings['interferometer/oskar_vis_filename'])
    if sky.num_sources == 1:
        settings['simulator/use_gpus'] = 'false'
        settings['simulator/max_sources_per_chunk'] = '2'
    #print(json.dumps(settings.to_dict(), indent=4))
    sim = oskar.Interferometer(settings=settings)
    sim.set_sky_model(sky)
    sim.set_telescope_model(tel)
    sim.run()


def make_diff_image_stats(filename1, filename2, use_w_projection,
                          out_image_root=None):
    """Make an image of the difference between two visibility data sets.

    This function assumes that the observation parameters for both data sets
    are identical. (It will fail horribly otherwise!)
    """
    # Set up an imager.
    imager = oskar.Imager(precision='double')
    imager.set(fov_deg=5.0, image_size=5000, fft_on_gpu=True)
    if out_image_root is not None:
        imager.output_root = out_image_root

    LOG.info("Imaging differences between '%s' and '%s'", filename1, filename2)
    (hdr1, handle1) = oskar.VisHeader.read(filename1)
    (hdr2, handle2) = oskar.VisHeader.read(filename2)
    block1 = oskar.VisBlock.create_from_header(hdr1)
    block2 = oskar.VisBlock.create_from_header(hdr2)
    if hdr1.num_blocks != hdr2.num_blocks:
        raise RuntimeError("'%s' and '%s' have different dimensions!" %
                           (filename1, filename2))
    if use_w_projection:
        imager.set(algorithm='W-projection')
        imager.coords_only = True
        for i_block in range(hdr1.num_blocks):
            block1.read(hdr1, handle1, i_block)
            imager.update_from_block(hdr1, block1)
        imager.coords_only = False
        imager.check_init()
        LOG.info("Using %d W-planes", imager.num_w_planes)
    for i_block in range(hdr1.num_blocks):
        block1.read(hdr1, handle1, i_block)
        block2.read(hdr2, handle2, i_block)
        block1.cross_correlations()[...] -= block2.cross_correlations()
        imager.update_from_block(hdr1, block1)
    del handle1, handle2, hdr1, hdr2, block1, block2

    # Finalise image and return it to Python.
    output = imager.finalise(return_images=1)
    image = output['images'][0]

    LOG.info("Generating image statistics")
    image_size = imager.image_size
    box_size = int(0.1 * image_size)
    centre = image[
        (image_size - box_size)//2:(image_size + box_size)//2,
        (image_size - box_size)//2:(image_size + box_size)//2]
    return {
        'image_medianabs': numpy.median(numpy.abs(image)),
        'image_mean': numpy.mean(image),
        'image_std': numpy.std(image),
        'image_rms': numpy.sqrt(numpy.mean(image**2)),
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
    elif metric_key == 'image_medianabs':
        metric_name = 'MEDIAN(ABS(image)) [Jy/beam]'
    sky_model = 'GLEAM + A-team'
    if single_source_offset >= 0:
        sky_model = 'single source'
    plt.title('%s for %s field (%s)' % (metric_name, field_name, sky_model))
    plt.xlabel('Element gain standard deviation [dB]')
    plt.ylabel('Element phase standard deviation [deg]')
    cbar = plt.colorbar(sp, ax=ax1)
    cbar.set_label('log10(%s)' % metric_name)
    plt.savefig('%s_%s_%s.png' % (prefix, field_name, metric_key))
    plt.close('all')


def run_single(prefix_field, settings, sky, tel,
               gain_std_dB, phase_std_deg, out0_name, results):
    """Run a single simulation and generate image statistics for it."""
    out = '%s_%.3f_dB_%.2f_deg' % (prefix_field, gain_std_dB, phase_std_deg)
    if out in results:
        LOG.info("Using cached results for '%s'", out)
        return
    out_name = out + '.vis'
    gain_std = numpy.power(10.0, gain_std_dB / 20.0) - 1.0
    tel.override_element_gains(1.0, gain_std)
    tel.override_element_phases(phase_std_deg)
    settings['interferometer/oskar_vis_filename'] = out_name
    make_vis_data(settings, sky, tel)
    out_image_root = None
    if out == 'GLEAM_100_MHz_EoR0_0.170_dB_1.20_deg':
        out_image_root = out
    use_w_projection = True
    if sky.num_sources == 1:
        use_w_projection = False
    results[out] = make_diff_image_stats(out0_name, out_name, use_w_projection,
                                         out_image_root)
    #os.remove(out_name)  # Delete visibility data to save space.


def run_set(prefix, base_settings, fields, axis_gain, axis_phase, specials,
            single_source_offset, plot_only):
    """Runs a set of simulations."""
    if not plot_only:
        # Load base telescope model.
        settings = oskar.SettingsTree('oskar_sim_interferometer')
        settings.from_dict(base_settings)
        tel = oskar.Telescope(settings=settings)

    # Iterate over fields.
    for field_name, field in fields.items():
        # Load result set, if it exists.
        prefix_field = prefix + '_' + field_name
        results = {}
        json_file = prefix_field + '_results.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as input_file:
                results = json.load(input_file)

        if not plot_only:
            # Update settings for field.
            settings_dict = base_settings.copy()
            settings_dict.update(field)
            settings.from_dict(settings_dict)
            ra_deg = float(settings['observation/phase_centre_ra_deg'])
            dec_deg = float(settings['observation/phase_centre_dec_deg'])
            length_sec = float(settings['observation/length'])
            settings['observation/start_time_utc'] = get_start_time(
                ra_deg, length_sec)
            tel.set_phase_centre(ra_deg, dec_deg)

            # Load or create the sky model.
            if single_source_offset >= 0:
                sky = oskar.Sky.from_array(numpy.array((
                    [ra_deg, dec_deg - single_source_offset, 1])))
                settings['interferometer/ignore_w_components'] = 'true'
            else:
                sky = oskar.Sky(settings=settings)
                num_sources0 = sky.num_sources
                append_bright_sources(sky)
                assert sky.num_sources - num_sources0 == 10
                settings['interferometer/ignore_w_components'] = 'false'

            # Simulate the 'perfect' case.
            tel.override_element_gains(1.0, 0.0)
            tel.override_element_phases(0.0)
            out0_name = '%s_no_errors.vis' % prefix_field
            settings['interferometer/oskar_vis_filename'] = out0_name
            make_vis_data(settings, sky, tel)

            # Simulate the error cases.
            for gain_std_dB in axis_gain:
                for phase_std_deg in axis_phase:
                    run_single(prefix_field, settings, sky, tel,
                               gain_std_dB, phase_std_deg, out0_name, results)

            # Simulate the 'special' error cases.
            for _, params in specials.items():
                gain_std_dB = params['gain_std_dB']
                phase_std_deg = params['phase_std_deg']
                run_single(prefix_field, settings, sky, tel,
                           gain_std_dB, phase_std_deg, out0_name, results)

        # Generate plot for the field.
        if single_source_offset >= 0:
            make_plot(prefix, field_name, 'image_centre_rms', results,
                      axis_gain, axis_phase, specials, single_source_offset)
        else:
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
            'use_gpus': 'true',
            'max_sources_per_chunk': '23000'
        },
        'observation' : {
            'start_frequency_hz': '100e6',
            'frequency_inc_hz': '100e3',
            'length': '21600.0',
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
            'max_time_samples_per_block': '4'
        }
    }

    # Define axes of parameter space.
    fields = {
        'EoR0': {
            'sky/oskar_sky_model/file': 'EoR0_20deg_point_source.osm',
            'observation/phase_centre_ra_deg': '0.0',
            'observation/phase_centre_dec_deg': '-27.0'
        },
        'EoR1': {
            'sky/oskar_sky_model/file': 'EoR1_20deg_point_source.osm',
            'observation/phase_centre_ra_deg': '60.0',
            'observation/phase_centre_dec_deg': '-30.0'
        },
        'EoR2': {
            'sky/oskar_sky_model/file': 'EoR2_20deg_point_source.osm',
            'observation/phase_centre_ra_deg': '170.0',
            'observation/phase_centre_dec_deg': '-10.0'
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
    plot_only = False
    single_source_offset = 2.25
    run_set('src1_100_MHz_%.2f_deg' % single_source_offset, base_settings,
            fields, axis_gain, axis_phase, specials, single_source_offset,
            plot_only)

    # GLEAM + A-team sky model simulations.
    run_set('GLEAM_100_MHz', base_settings,
            fields, axis_gain, axis_phase, specials, -1, plot_only)


if __name__ == '__main__':
    main()
